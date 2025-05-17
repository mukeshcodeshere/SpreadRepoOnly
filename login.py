import os

def get_user_credentials():
    """Prompt the user to input their username and password."""
    print("Please enter your login details.")
    username = input("Username: ")
    password = input("Password: ")
    return username, password

def create_or_update_env_file(username, password):
    """Create or update the .env file with the provided username and password."""
    env_content = f"USERNAME_LOGIN={username}\nPASSWORD_LOGIN={password}\n"
    
    try:
        # Write the credentials to the .env file
        with open("credential.env", "w") as f:
            f.write(env_content)
        print("Your credentials have been entered.")
    except Exception as e:
        print(f"An error occurred while entering the credentials: {e}")

def delete_env_file():
    """Delete the .env file."""
    if os.path.exists("credential.env"):
        try:
            os.remove("credential.env")
            print(".env file has been deleted for security.")
        except Exception as e:
            print(f"An error occurred while deleting the .env file: {e}")

def login():
    """Handle the login process and verify the user credentials."""
    successful_login = False
    
    while not successful_login:
        print("\n--- Login Process ---")
        username, password = get_user_credentials()

        # Save credentials in the .env file
        create_or_update_env_file(username, password)

        try:
            # Importing the necessary function after environment variables are set
            from gcc_sparta_lib import test_auth_data_pull

            # Attempt to pull data and check if the login is successful
            login_status = test_auth_data_pull()

            if login_status == "successful_login":
                print("\nLogin successful! Welcome.")
                successful_login = True  # Set the flag to True if login is successful
            else:
                print("\nLogin failed. Please double-check your credentials and try again.")
                delete_env_file()  # Delete the .env file if login fails
        except Exception as e:
            print(f"An error occurred while attempting to log in: {e}")
            delete_env_file()  # Clean up .env file in case of error
    
    return successful_login  # Return the flag indicating success
