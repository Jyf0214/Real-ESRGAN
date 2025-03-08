import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import torch

def main():
    """Inference demo for Real-ESRGAN (CPU专用版)"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='输入图像或文件夹')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('模型名称: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='输出目录')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help='去噪强度 [0,1] (仅适用于realesr-general-x4v3)')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='最终放大倍数')
    parser.add_argument('--model_path', type=str, default=None, help='[可选] 自定义模型路径')
    parser.add_argument('--suffix', type=str, default='out', help='输出文件后缀')
    parser.add_argument('-t', '--tile', type=int, default=0, help='分块大小 (0=禁用)')
    parser.add_argument('--tile_pad', type=int, default=10, help='分块填充')
    parser.add_argument('--pre_pad', type=int, default=0, help='预处理填充')
    parser.add_argument('--face_enhance', action='store_true', help='使用GFPGAN增强面部')
    parser.add_argument('--fp32', action='store_true', help='使用FP32精度')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='Alpha通道上采样器: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='输出格式: auto | jpg | png')
    
    args = parser.parse_args()

    # 模型初始化
    args.model_name = args.model_name.split('.')[0]
    model_mapping = {
        'RealESRGAN_x4plus': (RRDBNet(3,3,64,23,32,4), 4, ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']),
        'RealESRNet_x4plus': (RRDBNet(3,3,64,23,32,4), 4, ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']),
        'RealESRGAN_x4plus_anime_6B': (RRDBNet(3,3,64,6,32,4), 4, ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']),
        'RealESRGAN_x2plus': (RRDBNet(3,3,64,23,32,2), 2, ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']),
        'realesr-animevideov3': (SRVGGNetCompact(num_conv=16, upscale=4, act_type='prelu'), 4, ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']),
        'realesr-general-x4v3': (SRVGGNetCompact(num_conv=32, upscale=4, act_type='prelu'), 4, [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ])
    }
    
    model, netscale, file_url = model_mapping[args.model_name]

    # 模型路径处理
    if args.model_path is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
        os.makedirs(model_dir, exist_ok=True)
        model_paths = []
        for url in file_url:
            filename = os.path.basename(url)
            model_path = os.path.join(model_dir, filename)
            if not os.path.exists(model_path):
                model_path = load_file_from_url(url, model_dir=model_dir, progress=True)
            model_paths.append(model_path)
        args.model_path = model_paths[0] if len(model_paths) == 1 else model_paths

    # DNI权重处理
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        if isinstance(args.model_path, str):
            wdn_path = args.model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            args.model_path = [args.model_path, wdn_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # 初始化超分辨率器
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=args.model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32
    )
    upsampler.model.to('cpu')  # 强制使用CPU

    # 面部增强处理
    if args.face_enhance:
        from gfpgan import GFPGANer
        
        # GFPGAN模型路径
        gfpgan_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gfpgan', 'weights')
        os.makedirs(gfpgan_dir, exist_ok=True)
        gfpgan_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        gfpgan_path = os.path.join(gfpgan_dir, 'GFPGANv1.3.pth')
        if not os.path.exists(gfpgan_path):
            gfpgan_path = load_file_from_url(gfpgan_url, model_dir=gfpgan_dir, progress=True)
        
        # 初始化面部增强器
        face_enhancer = GFPGANer(
            model_path=gfpgan_path,
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )
        face_enhancer.face_enhancer.to('cpu')  # 强制使用CPU

    # 处理输入输出
    os.makedirs(args.output, exist_ok=True)
    input_paths = [args.input] if os.path.isfile(args.input) else sorted(glob.glob(os.path.join(args.input, '*')))
    
    for path in input_paths:
        imgname, ext = os.path.splitext(os.path.basename(path))
        print(f'Processing: {imgname}')
        
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
            
            # 保存结果
            ext = ext[1:] if args.ext == 'auto' else args.ext
            if img.shape[2] == 4:  # 带Alpha通道的强制保存为PNG
                ext = 'png'
            save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{ext}')
            cv2.imwrite(save_path, output)
            
        except Exception as e:
            print(f'Error processing {imgname}: {str(e)}')
            if "CUDA" in str(e):  # 友好的错误提示
                print("提示：请尝试使用 --tile 参数减少显存占用（即使使用CPU也应分块处理）")

if __name__ == '__main__':
    main()