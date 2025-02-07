export function initDataUploader() {
    // Get uploader elements
    const dataUploader = document.getElementById('dataUploaderContent');
    
    if (!dataUploader) {
      console.error('MoCap Data Uploader content element not found.');
      return;
    }
  
    const uploadForm = document.getElementById('uploadForm');
    const uploadStatus = document.getElementById('uploadStatus');
    const fileInput = document.getElementById('fileInput');
    const fileDropdown = document.getElementById('fileDropdown');
    const processButton = document.getElementById('processDataButton');
    const processStatus = document.getElementById('processStatus');
    const deleteButton = document.getElementById('deleteDataButton');
  
    let eventSource; 
  
    // Reset the uploader state
    function resetUploaderState(delay = 3000) {
      setTimeout(() => {
        fileInput.value = '';
        uploadStatus.textContent = '';
      }, delay);
    }
  
    // Refresh file list (mocap + video)
    function refreshFileList() {
      // Clear dropdown
      while (fileDropdown.firstChild) {
        fileDropdown.removeChild(fileDropdown.firstChild);
      }
  
      Promise.all([
        fetch('/api/data/mocap').then(response => {
          if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
          return response.json();
        }),
        fetch('/api/data/video').then(response => {
          if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
          return response.json();
        })
      ])

        .then(([mocapFiles, videoFiles]) => {
          // Combine, sort, and populate
          const allFiles = mocapFiles.concat(videoFiles).sort((a, b) => a.localeCompare(b));
          allFiles.forEach(file => {
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
  
      const files = Array.from(fileInput.files); 
      if (files.length === 0) {
        uploadStatus.textContent = 'Please select at least one file to upload.';
        resetUploaderState();
        return;
      }
  
      // Define possible video extensions
      const videoExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
  
      // Separate files by type
      const mocapFiles = files.filter(file =>
        file.name.toLowerCase().endsWith('.csv')
      );
      console.log("Uploading these files:", files.map(f => f.name));
      const videoFiles = files.filter(file => {
        const lowerName = file.name.toLowerCase();
        return videoExtensions.some(ext => lowerName.endsWith(ext));
      });
  
      const uploadPromises = [];
      let uploadMessage = "";
  
      // 1) Upload .csv files to /api/data/mocap using field name "files"
      if (mocapFiles.length > 0) {
        const mocapFormData = new FormData();
        mocapFiles.forEach(file => mocapFormData.append('files', file));
        
        uploadPromises.push(
          fetch('/api/data/mocap', {
            method: 'POST',
            body: mocapFormData,
          })
            .then(response => {
              if (!response.ok) {
                // attempt to parse error body
                return response.json().then(data => {
                  throw new Error(data.message || 'Failed to upload mocap files.');
                });
              }
              return response.json();
            })
            .then(data => {
              const uploadedFiles = data.uploadedFiles.map(f => f.filename).join(', ');
              const ignoredFiles = data.ignoredFiles.join(', ');
              uploadMessage += `Mocap Uploaded: ${uploadedFiles} ${ignoredFiles}. `;
            })
        );
      }
  
      // 2) Upload video files to /api/data/video using field name "files"
      if (videoFiles.length > 0) {
        const videoFormData = new FormData();
        videoFiles.forEach(file => videoFormData.append('files', file));
  
        uploadPromises.push(
          fetch('/api/data/video', {
            method: 'POST',
            body: videoFormData,
          })
            .then(response => {
              if (!response.ok) {
                return response.json().then(data => {
                  throw new Error(data.message || 'Failed to upload video files.');
                });
              }
              return response.json();
            })
            .then(data => {
              const uploadedFiles = data.uploadedFiles.map(f => f.filename).join(', ');
              const ignoredFiles = data.ignoredFiles.join(', ');
              uploadMessage += `Video Uploaded: ${uploadedFiles},${ignoredFiles}. `;
            })
        );
      }
  
      if (uploadPromises.length === 0) {
        uploadStatus.textContent = "No supported file types selected.";
        resetUploaderState();
        return;
      }
  
      // Wait for all uploads to finish
      Promise.all(uploadPromises)
        .then(() => {
          uploadStatus.textContent = uploadMessage;
          refreshFileList();
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
      const videoExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
      const selectedFiles = Array.from(fileDropdown.selectedOptions).map(option => option.value);
      console.log('Selected files:', selectedFiles);
  
      if (selectedFiles.length === 0) {
        processStatus.textContent = 'Please select files to process.';
        return;
      }
  
      resetProgressBar();
      processButton.disabled = true;
  
      if (eventSource) {
        eventSource.close();
        eventSource = null;
        console.log('Closed previous SSE connection');
      }
  
      processStatus.textContent = 'Processing data...';
  
      // Distinguish CSV vs. video files
      
      const mocapFiles = selectedFiles.filter(file =>
        file.toLowerCase().endsWith('.csv')
      );
      const videoFiles = selectedFiles.filter(file => {
        const lower = file.toLowerCase();
        return videoExtensions.some(ext => lower.endsWith(ext));
      });
  
      // Make requests for each type
      const processPromises = [];
  
      // Example: /api/process_mocap for CSV
      if (mocapFiles.length > 0) {
        processPromises.push(
          fetch('/api/process_mocap', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files: mocapFiles }),
          })
            .then(response => {
              if (!response.ok) {
                return response.json().then(data => {
                  throw new Error(data.message || 'Failed to process mocap files.');
                });
              }
              return response.json();
            })
        );
      }
  
      // Example: /api/process_video for videos
      if (videoFiles.length > 0) {
        processPromises.push(
          fetch('/api/process_video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files: videoFiles }),
          })
            .then(response => {
              if (!response.ok) {
                return response.json().then(data => {
                  throw new Error(data.message || 'Failed to process video files.');
                });
              }
              return response.json();
            })
        );
      }
  
      // When both requests finish, open SSE for progress
      Promise.all(processPromises)
        .then(() => {
          // Open the relevant SSE connection
          if (mocapFiles.length > 0) {
            eventSource = new EventSource('/api/process_mocap/progress');
          } else if (videoFiles.length > 0) {
            eventSource = new EventSource('/api/process_video/progress');
          }
  
          if (eventSource) {
            eventSource.onopen = () => {
              console.log('SSE connection opened');
            };
  
            eventSource.onmessage = (event) => {
              const progressUpdate = JSON.parse(event.data);
              updateProgressBar(progressUpdate);
            };
  
            eventSource.onerror = (error) => {
              console.error('SSE Error:', error);
              if (eventSource && eventSource.readyState === EventSource.CLOSED) {
                console.log('SSE connection closed by the server.');
              } else if (eventSource) {
                console.error('Unexpected SSE error occurred.');
                processStatus.textContent = 'Error receiving progress updates.';
              }
              if (eventSource) {
                eventSource.close();
                eventSource = null;
                console.log('EventSource connection closed due to error.');
              }
              processButton.disabled = false;
            };
          } else {
            processStatus.textContent = 'No processing event source available.';
            processButton.disabled = false;
          }
        })
        .catch(error => {
          console.error('Error processing data:', error);
          processStatus.textContent = `Error: ${error.message}`;
          processButton.disabled = false;
        });
    });
  
    // Handle deletion of selected files
    deleteButton.addEventListener('click', () => {
      const selectedFiles = Array.from(fileDropdown.selectedOptions).map(option => option.value);
  
      if (selectedFiles.length === 0) {
        processStatus.textContent = 'Please select files to delete.';
        return;
      }
  
      if (!confirm(`Are you sure you want to delete the selected files?`)) {
        return;
      }
  
      processStatus.textContent = 'Deleting files...';
  
      // Distinguish CSV vs. video
      const videoExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
      const mocapFilesToDelete = selectedFiles.filter(file =>
        file.toLowerCase().endsWith('.csv')
      );
      const videoFilesToDelete = selectedFiles.filter(file => {
        const lower = file.toLowerCase();
        return videoExtensions.some(ext => lower.endsWith(ext));
      });
  
      const deletePromises = [];
  
      if (mocapFilesToDelete.length > 0) {
        // e.g., DELETE /api/data/mocap
        deletePromises.push(
          fetch('/api/data/mocap', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files: mocapFilesToDelete }),
          })
            .then(response => {
              if (!response.ok) {
                return response.json().then(data => {
                  throw new Error(data.message || 'Failed to delete mocap files.');
                });
              }
              return response.json();
            })
        );
      }
  
      if (videoFilesToDelete.length > 0) {
        // e.g., DELETE /api/data/video
        deletePromises.push(
          fetch('/api/data/video', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files: videoFilesToDelete }),
          })
            .then(response => {
              if (!response.ok) {
                return response.json().then(data => {
                  throw new Error(data.message || 'Failed to delete video files.');
                });
              }
              return response.json();
            })
        );
      }
  
      Promise.all(deletePromises)
        .then(results => {
          let message = '';
          results.forEach(result => {
            if (result.status === 'success') {
              message += `Deleted files: ${result.deletedFiles.join(', ')}. `;
            } else if (result.status === 'partial_success') {
              message += `Deleted files: ${result.deletedFiles.join(', ')}. Failed to delete: ${result.failedFiles.join(', ')}. `;
            } else {
              message += `Error: ${result.message}. `;
            }
          });
          processStatus.textContent = message;
          refreshFileList();
        })
        .catch(error => {
          console.error('Error deleting files:', error);
          processStatus.textContent = `Error deleting files: ${error.message}`;
        });
    });
  
    // Reset the progress bar
    function resetProgressBar() {
      const progressBar = document.getElementById('progressBar');
      const processStatus = document.getElementById('processStatus');
      progressBar.style.width = '0%';
      processStatus.textContent = '';
      console.log('Progress bar and status reset');
    }
  
    // Update the progress bar from SSE
    function updateProgressBar(progressUpdate) {
      const progressBar = document.getElementById('progressBar');
      const processStatus = document.getElementById('processStatus');
      processButton.disabled = true;
  
      if (progressUpdate.currentFile) {
        processStatus.textContent = `Processing file: ${progressUpdate.currentFile}`;
        console.log('Processing', progressUpdate.progress);
      }
  
      if (progressUpdate.progress !== undefined) {
        progressBar.style.width = `${progressUpdate.progress}%`;
        processStatus.textContent = `Progress: ${progressUpdate.progress}% of all files. Currently processing: ${progressUpdate.currentFile}`;
        console.log(progressUpdate.progress);
      }
  
      if (progressUpdate.message === 'Processing complete!') {
        console.log('Processing complete! Closing SSE connection.');
        processStatus.textContent = 'Processing complete!';
        progressBar.style.width = '100%';
  
        if (eventSource) {
          eventSource.close();
          eventSource = null;
        }
        processButton.disabled = false;
      }
    }
  
    // Simple CSS for a loading bar animation
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
  
    // Initial file list fetch
    refreshFileList();
  }
  