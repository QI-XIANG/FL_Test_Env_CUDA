import os

def add_to_gitignore(scan_path, gitignore_path=".gitignore", max_file_size_mb=100):
    """
    Scans files and folders under the given path and appends to the .gitignore file
    with entries for files exceeding the max_file_size_mb limit.
    
    Args:
        scan_path (str): The path to scan for large files
        gitignore_path (str): The path to the .gitignore file
        max_file_size_mb (int): The maximum file size in MB
    """
    # Convert MB to bytes
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    ignore_list = []
    
    # Normalize the scan path
    scan_path = os.path.abspath(scan_path)
    
    try:
        # Read existing .gitignore entries if file exists
        existing_entries = set()
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as f:
                existing_entries = {line.strip() for line in f if line.strip() and not line.startswith('#')}

        # Scan directory tree
        for root, _, files in os.walk(scan_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if (os.path.isfile(file_path) and 
                        os.path.getsize(file_path) > max_file_size_bytes):
                        # Get relative path from the scan_path directory
                        relative_path = os.path.relpath(file_path, os.path.dirname(scan_path))
                        # Normalize path separators for gitignore
                        relative_path = relative_path.replace(os.sep, '/')
                        if relative_path not in existing_entries:
                            ignore_list.append(relative_path)
                except OSError as e:
                    print(f"Warning: Could not process {file_path}: {e}")

        # Append new entries if there are any
        if ignore_list:
            with open(gitignore_path, "a") as f:  # Changed from 'w' to 'a' to append
                # Add a header if file was empty/new
                if not existing_entries and os.stat(gitignore_path).st_size == 0:
                    f.write("# Auto-generated large file ignores\n")
                for item in ignore_list:
                    f.write(f"{item}\n")
                    print(f"Added '{item}' to .gitignore.")
        else:
            print("No new large files found to add to .gitignore.")

    except PermissionError as e:
        print(f"Error: Permission denied accessing {gitignore_path}: {e}")
    except IOError as e:
        print(f"Error: Failed to write to {gitignore_path}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Get current path
    current_path = os.getcwd()
    max_size = 100  # Maximum file size in MB
    add_to_gitignore(current_path, max_file_size_mb=max_size)