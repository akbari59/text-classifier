import json
import requests

BASE_URL = "http://127.0.0.1:5000/"

# You can use this function to test your api
# Make sure the server is running locally on BASE_URL`
# or change the hardcoded localhost above
def test_predict():
    """
    Test the predict route with test data
    """
    test_description = {"description": "this is a test description about Dementia"}
    print("Calling API with test description:")
    response = requests.post(f"{BASE_URL}/predict", data=json.dumps(test_description))
    print("Response: ")
    print(response.status_code)
    print(response.json())
    assert response.status_code == 200


if __name__ == "__main__":
    test_predict()