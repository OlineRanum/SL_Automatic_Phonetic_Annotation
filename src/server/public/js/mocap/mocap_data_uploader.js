export function initMoCapDataUploader() {
    const mocapDataUploader = document.getElementById('mocapDataUploaderContent');

    if (!mocapDataUploader) {
        console.error('MoCap Data Uploader content element not found.');
        return;
    }

    const uploadForm = document.getElementById('uploadForm');
    const uploadStatus = document.getElementById('uploadStatus');
    const fileInput = document.getElementById('fileInput');
    const fileDropdown = document.getElementById('fileDropdown');
    const processButton = document.getElementById('processDataButton');
    const processStatus = document.getElementById('processStatus');

    // Function to reset the uploader state
    function resetUploaderState(delay = 3000) {
        setTimeout(() => {
            fileInput.value = '';
            uploadStatus.textContent = '';
        }, delay);
    }

    // Function to refresh the file list and sort alphabetically
    function refreshFileList() {
        // Clear the current options
        fileDropdown.innerHTML = '';

        // Fetch and populate the new file list
        fetch('/api/data/mocap')
            .then(response => response.json())
            .then(files => {
                // Sort the file list alphabetically
                files.sort((a, b) => a.localeCompare(b));

                // Populate the dropdown with sorted files
                files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file;
                    option.textContent = file;
                    fileDropdown.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Error refreshing file list:', error);
                const errorOption = document.createElement('option');
                errorOption.textContent = 'Error loading files';
                fileDropdown.appendChild(errorOption);
            });
    }

    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
    
        const files = Array.from(fileInput.files); // Get all selected files
    
        if (files.length === 0) {
            uploadStatus.textContent = 'Please select at least one file to upload.';
            resetUploaderState();
            return;
        }
    
        const formData = new FormData();
        files.forEach(file => formData.append('mocapFile', file));
    
        fetch('/api/data/mocap', {
            method: 'POST',
            body: formData,
        })
            .then(response => response.json())
            .then(data => {
                const uploadedFiles = data.uploadedFiles.map(f => f.filename).join(', ');
                const ignoredFiles = data.ignoredFiles.join(', ');
                uploadStatus.textContent = `Uploaded: ${uploadedFiles}. Ignored (already exists): ${ignoredFiles}.`;
                refreshFileList(); // Refresh the file list after upload
                resetUploaderState();
            })
            .catch(error => {
                console.error('Error uploading files:', error);
                uploadStatus.textContent = 'Error uploading files.';
                resetUploaderState();
            });
    });
    

    processButton.addEventListener('click', () => {
        const selectedFiles = Array.from(fileDropdown.selectedOptions).map(option => option.value);

        if (selectedFiles.length === 0) {
            processStatus.textContent = 'Please select files to process.';
            return;
        }

        processStatus.textContent = 'Processing data...';

        // Simulate processing
        setTimeout(() => {
            processStatus.textContent = `Processing completed for: ${selectedFiles.join(', ')}`;
        }, 2000);
    });

    document.getElementById('deleteDataButton').addEventListener('click', () => {
        const selectedFiles = Array.from(fileDropdown.selectedOptions).map(option => option.value);
    
        if (selectedFiles.length === 0) {
            processStatus.textContent = 'Please select files to delete.';
            return;
        }
    
        // Confirm deletion
        if (!confirm(`Are you sure you want to delete the selected files?`)) {
            return;
        }
    
        processStatus.textContent = 'Deleting files...';
    
        // Send delete request to the server
        fetch('/api/data/mocap', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files: selectedFiles }),
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to delete files.');
                }
                return response.json();
            })
            .then(data => {
                processStatus.textContent = `Deleted files: ${data.deletedFiles.join(', ')}`;
                refreshFileList(); // Refresh the file list after deletion
            })
            .catch(error => {
                console.error('Error deleting files:', error);
                processStatus.textContent = 'Error deleting files.';
            });
    });
    

    // Initial file list population
    refreshFileList();
}
