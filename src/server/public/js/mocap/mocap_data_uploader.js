export function initMoCapDataUploader() {
    const mocapDataUploader = document.getElementById('mocapDataUploaderContent');

    if (!mocapDataUploader) {
        console.error('MoCap Data Uploader content element not found.');
        return;
    }

    // Handle file upload
    const uploadForm = document.getElementById('uploadForm');
    const uploadStatus = document.getElementById('uploadStatus');
    const fileInput = document.getElementById('fileInput');

    // Function to reset the form and states with a delay
    function resetUploaderState(delay = 3000) {
        setTimeout(() => {
            fileInput.value = ''; // Clear file input
            uploadStatus.textContent = ''; // Clear upload status
        }, delay);
    }

    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();

        const file = fileInput.files[0]; // Get the selected file

        if (!file) {
            uploadStatus.textContent = 'Please select a file to upload.';
            resetUploaderState(); // Reset after 3 seconds
            return;
        }

        const formData = new FormData();
        formData.append('mocapFile', file);

        // Perform the upload
        fetch('/api/data/mocap', {
            method: 'POST',
            body: formData,
        })
            .then((response) => {
                if (response.status === 409) {
                    // File already exists
                    uploadStatus.textContent = `File "${file.name}" already exists in repository.`;
                    throw new Error('File already exists in repository.');
                }
                if (!response.ok) {
                    // General upload error
                    throw new Error('File upload failed.');
                }
                return response.json();
            })
            .then((data) => {
                uploadStatus.textContent = `File uploaded successfully: ${data.filename}`;
                resetUploaderState(); // Reset after 3 seconds
            })
            .catch((error) => {
                console.error('Error uploading file:', error);

                // Ensure error messages display properly
                if (error.message === 'File already exists in repository.') {
                    uploadStatus.textContent = `File "${file.name}" already exists in repository.`;
                } else {
                    uploadStatus.textContent = 'Error uploading file.';
                }

                // Keep the error message visible for 3 seconds before resetting
                setTimeout(() => {
                    uploadStatus.textContent = '';
                    fileInput.value = ''; // Clear file input after message timeout
                }, 3000);
            });
    });

    // Reset the form on navigation or UI refresh
    resetUploaderState(0); // No delay for initial reset
}
