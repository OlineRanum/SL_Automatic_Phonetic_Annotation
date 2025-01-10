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
    const deleteButton = document.getElementById('deleteDataButton'); // Ensure this element exists

    let eventSource; 

    // Function to reset the uploader state
    function resetUploaderState(delay = 3000) {
        setTimeout(() => {
            fileInput.value = '';
            uploadStatus.textContent = '';
        }, delay);
    }

    // Function to refresh the file list and sort alphabetically
    function refreshFileList() {
        // Clear the current options to prevent duplication
        while (fileDropdown.firstChild) {
            fileDropdown.removeChild(fileDropdown.firstChild);
        }
    
        // Fetch and populate the new file list
        fetch('/api/data/mocap')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
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
    
    // Handle file uploads
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
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.message || 'Failed to upload files.');
                    });
                }
                return response.json();
            })
            .then(data => {
                const uploadedFiles = data.uploadedFiles.map(f => f.filename).join(', ');
                const ignoredFiles = data.ignoredFiles.join(', ');
                uploadStatus.textContent = `Uploaded: ${uploadedFiles}. \nIgnored (already exists): ${ignoredFiles}.`;
                refreshFileList(); // Refresh the file list after upload
                resetUploaderState();
            })
            .catch(error => {
                console.error('Error uploading files:', error);
                uploadStatus.textContent = `Error uploading files: ${error.message}`;
                resetUploaderState();
            });
    });
    
    // Handle processing of selected files
    processButton.addEventListener('click', () => {
        const selectedFiles = Array.from(fileDropdown.selectedOptions).map(option => option.value);
        console.log('Selected files:', selectedFiles);

        if (selectedFiles.length === 0) {
            processStatus.textContent = 'Please select files to process.';
            return;
        }

        resetProgressBar(); // Reset progress UI

        // Disable the process button to prevent multiple clicks
        processButton.disabled = true;

        // If a previous EventSource exists, close it before starting a new one
        if (eventSource) {
            eventSource.close();
            eventSource = null;
            console.log('Closed previous SSE connection'); // Debug log
        }

        processStatus.textContent = 'Processing data...';

        fetch('/api/process_mocap', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files: selectedFiles }),
        })
            .then(response => {
                if (!response.ok) {
                    // Attempt to parse the error response as JSON
                    return response.json().then(data => {
                        throw new Error(data.message || 'Unknown error');
                    }).catch(() => {
                        // If parsing fails, throw a generic error
                        throw new Error('An error occurred while processing the request.');
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    // Initialize a new EventSource connection
                    eventSource = new EventSource('/api/process_mocap/progress');
                    console.log('Opened new SSE connection'); // Debug log

                    eventSource.onopen = () => {
                        console.log('SSE connection opened');
                    };

                    eventSource.onmessage = (event) => {
                        const progressUpdate = JSON.parse(event.data);
                        // console.log('Progress Update Received:', progressUpdate); // Debug log
                        updateProgressBar(progressUpdate);
                    };

                    eventSource.onerror = (error) => {
                        console.error('SSE Error:', error);
                    
                        // Check if the EventSource is already closed or null
                        if (eventSource && eventSource.readyState === EventSource.CLOSED) {
                            console.log('SSE connection closed by the server.');
                        } else if (eventSource) {
                            console.error('Unexpected SSE error occurred.');
                            processStatus.textContent = 'Error receiving progress updates.';
                        }
                    
                        // Ensure the EventSource is closed and nullified
                        if (eventSource) {
                            eventSource.close();
                            eventSource = null;
                            console.log('EventSource connection closed due to error.');
                        }
                        processButton.disabled = false; // Re-enable the button
                    };
                    
                } else {
                    processStatus.textContent = `Error: ${data.message}`;
                    processButton.disabled = false; // Re-enable the button
                }
            })
            .catch(error => {
                console.error('Error processing data:', error);
                processStatus.textContent = `Error: ${error.message}`;
                processButton.disabled = false; // Re-enable the button
            });
    });
    
    // Handle deletion of selected files
    deleteButton.addEventListener('click', () => {
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
    
        // Send delete request to the new delete endpoint
        fetch('/api/data/mocap/delete', { // Updated endpoint
            method: 'DELETE', // Updated method
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files: selectedFiles }),
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.message || `HTTP error! status: ${response.status}`);
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    processStatus.textContent = `Deleted files: ${data.deletedFiles.join(', ')}`;
                } else if (data.status === 'partial_success') {
                    processStatus.textContent = `Deleted files: ${data.deletedFiles.join(', ')}. Failed to delete: ${data.failedFiles.join(', ')}`;
                } else {
                    processStatus.textContent = `Error: ${data.message}`;
                }
                refreshFileList(); // Refresh the file list after deletion
            })
            .catch(error => {
                console.error('Error deleting files:', error);
                processStatus.textContent = `Error deleting files: ${error.message}`;
            });
    });

    // Function to reset the progress bar
    function resetProgressBar() {
        const progressBar = document.getElementById('progressBar');
        const processStatus = document.getElementById('processStatus');
        progressBar.style.width = '0%'; // Reset width
        processStatus.textContent = ''; // Clear status
        console.log('Progress bar and status reset'); // Debug log
    }

    // Function to update the progress bar based on progress updates
    function updateProgressBar(progressUpdate) {
        const progressBar = document.getElementById('progressBar');
        const processStatus = document.getElementById('processStatus');
        processButton.disabled = true; // Ensure the button remains disabled during processing
    
        if (progressUpdate.currentFile) {
            //console.log('Processing file:', progressUpdate.currentFile);
            processStatus.textContent = `Processing file: ${progressUpdate.currentFile}`;
            console.log('starting ', progressUpdate.progress); // Debug log
        }
    
        if (progressUpdate.progress !== undefined) {
            //console.log('Progress:', progressUpdate.progress);
            progressBar.style.width = `${progressUpdate.progress}%`;
            processStatus.textContent = `Progress: ${progressUpdate.progress}% for ${progressUpdate.currentFile}`;
            console.log(progressUpdate.progress); // Debug log
        }
    
        if (progressUpdate.message === 'Processing complete!') {
            console.log('Processing complete! Closing SSE connection.');
            processStatus.textContent = 'Processing complete!';
            progressBar.style.width = '100%';
            console.log('finished ', progressUpdate.progress); // Debug log


            // Close EventSource if it exists
            if (eventSource) {
                eventSource.close();
                eventSource = null; // Clear the reference
            }
            processButton.disabled = false; // Re-enable the button
            progressUpdate.progress = 0; // Reset progress
            progressUpdate.message = ''; // Clear message
            progressUpdate.currentFile = '';
        }
    }

    // Add loading animation CSS
    const style = document.createElement('style');
    style.textContent = `
    @keyframes loadingAnimation {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    #loadingBar {
        background: linear-gradient(90deg, #007bff, #00c4ff, #007bff);
        background-size: 200% 100%;
    }
    `;
    document.head.appendChild(style);
    
    // Initial file list population
    refreshFileList();
}
