import argparse
import os
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True, help='HF model directory (saved by Trainer)')
    ap.add_argument('--llama_cpp_path', required=True, help='Path to llama.cpp repo')
    ap.add_argument('--out', required=True, help='Output GGUF file path')
    ap.add_argument('--ftype', default='q8_0', help='Float/quant type for conversion (e.g., f16, f32, q8_0)')
    args = ap.parse_args()

    conv_py = os.path.join(args.llama_cpp_path, 'convert-hf-to-gguf.py')
    if not os.path.exists(conv_py):
        raise FileNotFoundError(f'convert-hf-to-gguf.py not found at {conv_py}')

    cmd = [
        'python', conv_py,
        '--model', args.model_dir,
        '--outfile', args.out,
        '--ftype', args.ftype
    ]
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd, cwd=args.llama_cpp_path)
    print('GGUF written to', args.out)


if __name__ == '__main__':
    main()
