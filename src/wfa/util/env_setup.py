import os
from pathlib import Path
from dotenv import load_dotenv
import ee

def setup_env(env_path: str | None = None):
    # Load .env into os.environ (optional path)
    if env_path:
        env_file = Path(env_path)
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
        else:
            load_dotenv()
    else:
        load_dotenv()

    # 1) Ensure ~/.cdsapirc exists from env (so cdsapi works headlessly)
    cds_url = os.getenv("CDSAPI_URL")
    cds_key = os.getenv("CDSAPI_KEY")
    cds_uid = os.getenv("CDSAPI_UID")
    cds_rc = Path.home() / ".cdsapirc"
    if cds_url and cds_key and not cds_rc.exists():
        cds_rc.write_text(f"url: {cds_url}\nkey: {cds_key}\n")

    # 2) Optionally initialize Earth Engine service account (if env is set)
    if os.getenv("EE_USE_SERVICE_ACCOUNT", "").lower() in {"1", "true", "yes"}:
        # try:
        import ee
        from google.oauth2 import service_account
        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        proj = os.getenv("EE_PROJECT")
        if sa_path and Path(sa_path).exists():
            creds = service_account.Credentials.from_service_account_file(
                sa_path,
                scopes=[
                    "https://www.googleapis.com/auth/earthengine",
                    "https://www.googleapis.com/auth/cloud-platform",
                ],
            )
            ee.Initialize(credentials=creds, project=proj)

            # print all environment variables
            print('CDSAPI_URL:', os.environ["CDSAPI_URL"])
            print('CDSAPI_KEY:', os.environ["CDSAPI_KEY"])
            print('GOOGLE_APPLICATION_CREDENTIALS:', os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
            print('EE_PROJECT:', os.environ["EE_PROJECT"])

        # except Exception:
        #     print("Error initializing Earth Engine service account.")
    
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load .env and configure CDS + Earth Engine.")
    parser.add_argument("--env", default=".env", help="Path to .env file (default: .env)")
    args = parser.parse_args()
    setup_env(args.env)
    print("Environment configured.")

if __name__ == "__main__":
    main()