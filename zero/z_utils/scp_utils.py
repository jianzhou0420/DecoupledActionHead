import paramiko
import os
import shutil  # Import shutil for rmtree


def scp_folder_recursive(sftp_client, local_dir, remote_dir):
    """
    Recursively copies a local directory and its contents to a remote directory.

    Args:
        sftp_client (paramiko.SFTPClient): An active SFTPClient instance.
        local_dir (str): The path to the local directory to copy.
        remote_dir (str): The destination path on the remote machine.
    """
    print(f"Starting recursive transfer of '{local_dir}' to '{remote_dir}'...")

    for item in os.listdir(local_dir):
        local_path = os.path.join(local_dir, item)
        remote_path = os.path.join(remote_dir, item)

        if os.path.isfile(local_path):
            try:
                sftp_client.put(local_path, remote_path)
                print(f"  Copied file: {local_path} -> {remote_path}")
            except Exception as e:
                print(f"  Error copying file {local_path}: {e}")
        elif os.path.isdir(local_path):
            try:
                # Create the directory on the remote server if it doesn't exist
                sftp_client.mkdir(remote_path)
                print(f"  Created directory: {remote_path}")
            except IOError as e:
                # Handle case where directory already exists (common)
                if "File exists" in str(e):
                    print(f"  Directory already exists: {remote_path}")
                else:
                    print(f"  Error creating directory {remote_path}: {e}")
            except Exception as e:
                print(f"  Error creating directory {remote_path}: {e}")

            # Recursively call for subdirectories
            scp_folder_recursive(sftp_client, local_path, remote_path)


def scp_to_another_computer(local_path, remote_path, hostname, username, password=None, key_filename=os.path.expanduser('~/.ssh/id_rsa'), port=22):
    """
    Connects to a remote computer and initiates SCP transfer for files or folders.
    """
    transport = None
    try:
        # Check if the local path actually exists before proceeding
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local path not found: {local_path}")

        transport = paramiko.Transport((hostname, port))

        if password:
            transport.connect(username=username, password=password)
        elif key_filename:
            try:
                k = paramiko.RSAKey.from_private_key_file(key_filename)
            except paramiko.PasswordRequiredException:
                key_password = input(f"Enter passphrase for key '{key_filename}': ")
                k = paramiko.RSAKey.from_private_key_file(key_filename, password=key_password)
            transport.connect(username=username, pkey=k)
        else:
            raise ValueError("Either 'password' or 'key_filename' must be provided for authentication.")

        sftp = paramiko.SFTPClient.from_transport(transport)

        if os.path.isfile(local_path):
            print(f"Copying single file '{local_path}' to '{username}@{hostname}:{remote_path}'...")
            sftp.put(local_path, remote_path)
            print("File copied successfully!")
        elif os.path.isdir(local_path):
            # Ensure the base remote directory exists before starting recursive copy
            try:
                sftp.mkdir(remote_path)
                print(f"Created base remote directory: {remote_path}")
            except IOError as e:
                if "File exists" in str(e):
                    print(f"Base remote directory already exists: {remote_path}")
                else:
                    raise  # Re-raise other IOErrors

            scp_folder_recursive(sftp, local_path, remote_path)
            print(f"Folder '{local_path}' copied successfully to '{username}@{hostname}:{remote_path}'!")
        else:
            print(f"Error: Local path '{local_path}' is neither a file nor a directory.")

    except paramiko.AuthenticationException:
        print("Authentication failed. Please check your username, password, or private key.")
    except paramiko.SSHException as e:
        print(f"SSH error: {e}")
    except FileNotFoundError as e:  # Catch the specific FileNotFoundError here
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if transport and transport.is_active():
            transport.close()


if __name__ == "__main__":
    print("\n--- Folder Transfer Example ---")
    print(f"Current working directory: {os.getcwd()}")

    # Define the base directory for the local folder
    # Using os.path.abspath ensures we get the full path
    local_folder_base = os.path.abspath("my_local_folder_for_scp_test")
    remote_destination_folder = "/tmp/my_remote_folder_copy"  # Remote destination path for the folder

    # Create the dummy local folder with subdirectories and files
    try:
        os.makedirs(os.path.join(local_folder_base, "subfolder1"), exist_ok=True)
        os.makedirs(os.path.join(local_folder_base, "subfolder2"), exist_ok=True)
        with open(os.path.join(local_folder_base, "file1.txt"), "w") as f:
            f.write("Content of file1.\n")
        with open(os.path.join(local_folder_base, "subfolder1", "file2.log"), "w") as f:
            f.write("Log content.\n")
        with open(os.path.join(local_folder_base, "subfolder2", "image.jpg"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x00IEND\xaeB`\x82")  # A tiny PNG representation
        print(f"Successfully created local test folder: {local_folder_base}")

        scp_to_another_computer(
            local_path=local_folder_base,  # Use the absolute path here
            remote_path=remote_destination_folder,
            hostname="127.0.0.1",  # Replace with your remote host IP or hostname
            username="jian",     # Replace with your remote username
        )

    except Exception as e:
        print(f"An error occurred during local folder creation or SCP: {e}")
    finally:
        # Clean up local test folder
        if os.path.exists(local_folder_base):
            print(f"Cleaning up local test folder: {local_folder_base}")
            shutil.rmtree(local_folder_base)
        else:
            print(f"Local test folder '{local_folder_base}' not found for cleanup.")
