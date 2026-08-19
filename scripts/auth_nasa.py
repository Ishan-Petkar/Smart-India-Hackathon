import pickle
import asf_search as asf

ASF_USER = "ishanpetkar"
ASF_PASS = "Petkar@99222"

def main():
    print("🌍 Authenticating with NASA ASF over current network...")
    try:
        session = asf.ASFSession().auth_with_creds(ASF_USER, ASF_PASS)
        with open("nasa_cookies.pkl", "wb") as f:
            pickle.dump(session.cookies, f)
        print("✅ SUCCESS! Authentication cookies securely saved to disk.")
        print("👉 You can now switch to your fast Wi-Fi and run the download script!")
    except Exception as e:
        print(f"❌ Auth failed: {e}")

if __name__ == "__main__":
    main()
