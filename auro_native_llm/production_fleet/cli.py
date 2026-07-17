from __future__ import annotations
import argparse, json
from .runtime import NovaRuntime

def main() -> None:
    parser=argparse.ArgumentParser(description="NOVA-governed Auro model fleet")
    parser.add_argument("message")
    parser.add_argument("--execute",action="store_true",help="approve bounded action proposals for a separate executor")
    args=parser.parse_args()
    print(json.dumps(NovaRuntime().respond(args.message,execute=args.execute),indent=2,ensure_ascii=False))

if __name__ == "__main__": main()

