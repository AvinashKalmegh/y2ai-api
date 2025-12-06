import requests

resp = requests.get(
    "https://api.linkedin.com/v2/me",
    headers={"Authorization": f"Bearer {os.getenv('LINKEDIN_ACCESS_TOKEN')}"}
)
print(resp.status_code, resp.text)
