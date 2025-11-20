import argparse
from huggingface_hub import HfApi, create_repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--repo_id', required=True)
    ap.add_argument('--private', action='store_true')
    args = ap.parse_args()

    api = HfApi()
    create_repo(args.repo_id, exist_ok=True, private=args.private)
    api.upload_folder(folder_path=args.model_dir, repo_id=args.repo_id)
    print('Uploaded', args.model_dir, 'to', args.repo_id)


if __name__ == '__main__':
    main()
