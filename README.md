# MirrorMe

Welcome to MirrorMe, a platform where you can virtually try on cloths on yourself.

# 🚀 Features

- Virtual Tryon for Indian Cloths.
- Background Change
- Skin Tone Compatibility
- Size recommendation
- Measurement Tips

Clone the Repo:

```
git clone <repository-url>
cd MirrorMe
```

To run the app:

``` 
python -m venv venv
source ./bin/activate
pip install -r requirements.txt
python app.py
```

The endpoints can be accessed via Swagger: http://localhost:8000/docs   


Endpoints:

**Auth**
- `POST /sign_up`: Enter your details to sign up
- `POST /request_otp`: Once signed in, enter the email to get an OTP
- `POST /verify_otp`: Enter the received OTP to verify the account
-  `POST /login`: Enter the credentails to login and authorize.

**Body Size Detection**

- ` POST /body_size/detect`: Upload your image and get your body size (XL, L, M, S, XS)
- `POST /body_size/detect`: Enter your physiological measurements and get your body size.
- ` POST /body_size/replace_background`: Upload your image and either upload a background image, enter a prompt or both to replace the background of your image.

**Jobs**

- `POST /create_job`:
    - *person_image*: upload the person's image
    - *overlay cloths*: upload one or more cloths to be overlay
    - *overlay_model_types*: for each cloth uploaded, specify if you want the cloth to cover full-body, upper-body, or lower-body.
    - *overlay_garment_types*: for each cloth uploaded, write the name of the cloth.
    - *overlay_styles*: for each cloth uploaded, mention what style that cloth is to be worn in. For example, Indian traditional, Western, Casual etc.
    - *intstuctions*: Any additional instruction you want to specify.

- `GET /jobs/next`: this endpoint is auto called in the code, nothing to be done here.
- `GET /jobs/{job_id}`: get details about a specific job by giving the job_id.
- `GET /get_all_jobs`: get all the jobs created by the user.
- `DELETE /delete_jobs`: delete all the jobs created by the user.

# 👤 Author

**Saurav Kumar**