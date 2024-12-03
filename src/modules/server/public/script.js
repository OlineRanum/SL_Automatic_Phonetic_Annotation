// public/script.js

document.addEventListener('DOMContentLoaded', () => {
    "use strict";

    // **GIF Viewer Elements**
    const gifSelect = document.getElementById('gif-select');
    const gifImage = document.getElementById('gif-image');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const togglePlayBtn = document.getElementById('toggle-play-btn'); // Toggle Button
    const frameInfo = document.getElementById('frame-info');
    const frameRateInput = document.getElementById('frame-rate');
    const loadingIndicator = document.getElementById('loading');
    const progressBar = document.getElementById('progress-bar');

    // **Reference Poses Viewer Elements**
    const poseSelectContainer = document.getElementById('pose-select-container');
    const referenceImageContainer = document.getElementById('reference-poses-container');
    const poseLoading = document.getElementById('pose-loading');
    const clearAllBtn = document.getElementById('clear-all-btn');

    // **Note-Taking Elements**
    const noteInput = document.getElementById('note-input');
    const saveNoteBtn = document.getElementById('save-note-btn');
    const notesList = document.getElementById('notes-list');

    // **All Notes Elements**
    const allNotesList = document.getElementById('all-notes-list');
    const updateAllNotesBtn = document.getElementById('update-all-notes-btn'); // Select the Update Button

    // **Notification Element**
    const notification = document.getElementById('notification');

    // **Current GIF and Frame Names**
    let currentGifName = '';
    let currentFrameName = '';

    // **GIF Viewer Variables**
    let currentFrames = [];
    let preloadedImages = [];
    let currentFrameIndex = 0;
    let isPlaying = false;
    let playInterval = null;
    let frameRate = 200; // Set to 200ms per frame to match 5 fps
    const preloadBatchSize = 10; // Number of frames to preload at a time

    // **Reference Poses Viewer Variables**
    let referencePoses = [];
    let sbReferenceFiles = []; // List of sb_references JPG files

    // **Fetch and Populate the GIF Dropdown**
    function fetchGifs() {
        fetch('/api/gifs', { cache: 'no-store' })
            .then(response => response.json())
            .then(gifs => {
                if (gifs.length === 0) {
                    gifSelect.innerHTML = '<option value="">No GIFs available</option>';
                    return;
                }
                // Clear existing options except the first
                gifSelect.innerHTML = '<option value="">--Select a GIF--</option>';
                gifs.forEach(gif => {
                    const option = document.createElement('option');
                    option.value = gif;
                    option.textContent = gif;
                    gifSelect.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Error fetching GIFs:', error);
                gifSelect.innerHTML = '<option value="">Error loading GIFs</option>';
            });
    }

    // **Fetch and Populate the Reference Poses Checkboxes**
    function fetchReferencePoses() {
        fetch('/api/reference_poses', { cache: 'no-store' })
            .then(response => response.json())
            .then(pngFiles => {
                if (pngFiles.length === 0) {
                    poseSelectContainer.innerHTML = '<p>No Reference Poses available</p>';
                    return;
                }
                referencePoses = pngFiles;
                pngFiles.forEach(png => {
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.id = `pose-${png}`;
                    checkbox.value = png;
                    checkbox.setAttribute('aria-label', `Select ${png} Pose`); // Accessibility

                    const label = document.createElement('label');
                    label.htmlFor = `pose-${png}`;
                    label.textContent = png;

                    // Create thumbnail image
                    const thumbnail = document.createElement('img');
                    thumbnail.src = `/reference_poses/${png}`;
                    thumbnail.alt = `${png} thumbnail`;
                    thumbnail.style.width = '50px'; // Adjust size as needed
                    thumbnail.style.height = 'auto';
                    thumbnail.style.marginRight = '10px';
                    thumbnail.style.border = '1px solid #ccc';
                    thumbnail.style.objectFit = 'cover';
                    thumbnail.style.flexShrink = '0'; // Prevent shrinking

                    const container = document.createElement('div');
                    container.style.display = 'flex';
                    container.style.alignItems = 'center';
                    container.style.marginBottom = '10px';

                    checkbox.style.marginRight = '10px'; // Space between checkbox and thumbnail

                    container.appendChild(checkbox);
                    container.appendChild(thumbnail);
                    container.appendChild(label);

                    poseSelectContainer.appendChild(container);

                    // Event Listener for checkbox
                    checkbox.addEventListener('change', (event) => {
                        if (event.target.checked) {
                            displayReferencePose(png);
                        } else {
                            removeReferencePose(png);
                        }
                        updateClearAllButtonState(); // Update Clear All button state
                    });
                });
            })
            .catch(error => {
                console.error('Error fetching Reference Poses:', error);
                poseSelectContainer.innerHTML = '<p>Error loading Reference Poses</p>';
            });
    }

    // **Fetch sb_references JPG Files**
    function fetchSbReferences() {
        fetch('/api/sb_references', { cache: 'no-store' })
            .then(response => response.json())
            .then(jpgFiles => {
                sbReferenceFiles = jpgFiles;
            })
            .catch(error => {
                console.error('Error fetching sb_references:', error);
                // Handle error if necessary
            });
    }

    // **Handle GIF Selection**
    gifSelect.addEventListener('change', () => {
        const selectedGif = gifSelect.value;
        if (selectedGif) {
            currentGifName = selectedGif; // Update current GIF name
            // Reset playback state
            stopPlayback();

            // Show loading indicator
            loadingIndicator.style.display = 'block';
            loadingIndicator.innerHTML = '<span>Loading frames...</span><div id="progress-container"><div id="progress-bar"></div></div>';
            progressBar.style.width = '0%';

            fetch(`/api/gifs/${encodeURIComponent(selectedGif)}/frames`, { cache: 'no-store' })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Frames not found.');
                    }
                    return response.json();
                })
                .then(frames => {
                    if (frames.length === 0) {
                        throw new Error('No frames available for this GIF.');
                    }
                    currentFrames = frames;
                    currentFrameIndex = 0;
                    preloadedImages = new Array(currentFrames.length).fill(null); // Initialize array
                    preloadBatch(currentFrameIndex);
                })
                .catch(error => {
                    console.error('Error fetching frames:', error);
                    loadingIndicator.innerHTML = 'Error loading frames.';
                });
        } else {
            // Reset if no GIF is selected
            currentGifName = '';
            currentFrames = [];
            preloadedImages = [];
            currentFrameIndex = 0;
            gifImage.src = '';
            frameInfo.textContent = 'Frame: 0';
            updateControls();
            togglePlayBtn.disabled = true; // Disable toggle when no GIF is selected
            loadingIndicator.style.display = 'none';
            progressBar.style.width = '0%';
            // Clear notes section
            clearNotesSection();
        }
    });

    // **Clear All Button Functionality**
    clearAllBtn.addEventListener('click', () => {
        clearAllSelectedPoses();
    });

    // **Preload a batch of frames starting from a specific index**
    function preloadBatch(startIndex) {
        const endIndex = Math.min(startIndex + preloadBatchSize, currentFrames.length);
        const promises = [];
        let loadedCount = 0;
        const totalToLoad = endIndex - startIndex;

        for (let i = startIndex; i < endIndex; i++) {
            if (!preloadedImages[i]) {
                const img = new Image();
                img.src = currentFrames[i]; // Adjust as per cache busting strategy
                const promise = new Promise((resolve, reject) => {
                    img.onload = () => {
                        preloadedImages[i] = img;
                        loadedCount++;
                        updateProgress((loadedCount / totalToLoad) * 100);
                        resolve();
                    };
                    img.onerror = () => {
                        console.error(`Failed to load frame ${i + 1}: ${currentFrames[i]}`);
                        loadedCount++;
                        updateProgress((loadedCount / totalToLoad) * 100);
                        resolve(); // Resolve even on error to continue
                    };
                });
                promises.push(promise);
            } else {
                loadedCount++;
                updateProgress((loadedCount / totalToLoad) * 100);
            }
        }

        Promise.all(promises)
            .then(() => {
                updateFrame();
                updateControls();
                // Preload next batch if nearing the end
                if (endIndex < currentFrames.length) {
                    preloadBatch(endIndex);
                }
                // Hide loading indicator if all frames are loaded
                if (preloadedImages.every(img => img !== null)) {
                    loadingIndicator.style.display = 'none';
                    updateProgress(0); // Reset progress bar
                }
            })
            .catch(preloadError => {
                console.error('Error preloading frames:', preloadError);
                loadingIndicator.innerHTML = 'Error loading frames.';
                updateProgress(0); // Reset progress bar
            });
    }

    // **Function to update progress bar**
    function updateProgress(percent) {
        progressBar.style.width = `${percent}%`;
    }

    // **Display Reference Pose and Corresponding sb_reference Image**
    function displayReferencePose(png) {
        // Check if the image already exists to prevent duplicates
        if (document.getElementById(`pose-wrapper-${png}`)) {
            return;
        }

        // Create a wrapper div
        const wrapper = document.createElement('div');
        wrapper.className = 'reference-image-wrapper';
        wrapper.id = `pose-wrapper-${png}`;

        // **Extract Base Name for sb_references**
        const baseName = png.replace(/^final_/, '').replace(/\.png$/i, '');
        const sbReferenceName = `${baseName}.jpg`;

        // **Check if sb_reference exists**
        const sbRefExists = sbReferenceFiles.includes(sbReferenceName);

        // **Create Caption for Pose PNG**
        const poseCaption = document.createElement('div');
        poseCaption.className = 'caption';
        poseCaption.textContent = png;

        // **Create Pose Image Element**
        const poseImg = document.createElement('img');
        poseImg.src = `/reference_poses/${png}`;
        poseImg.alt = png;

        poseImg.onload = () => {
            // Image loaded successfully
        };

        poseImg.onerror = () => {
            console.error(`Failed to load Reference Pose: ${png}`);
            poseImg.alt = 'Failed to load image.';
            // Optionally, you can display a placeholder image here
        };

        // **Append Pose Caption and Image to Wrapper**
        wrapper.appendChild(poseCaption);
        wrapper.appendChild(poseImg);

        // **If sb_reference exists, append its Caption and Image**
        if (sbRefExists) {
            const sbCaption = document.createElement('div');
            sbCaption.className = 'caption';
            sbCaption.textContent = sbReferenceName;

            const sbImg = document.createElement('img');
            sbImg.src = `/sb_references/${sbReferenceName}`;
            sbImg.alt = sbReferenceName;

            sbImg.onload = () => {
                // Image loaded successfully
            };

            sbImg.onerror = () => {
                console.error(`Failed to load sb_reference Image: ${sbReferenceName}`);
                sbImg.alt = 'Failed to load image.';
                // Optionally, you can display a placeholder image here
            };

            wrapper.appendChild(sbCaption);
            wrapper.appendChild(sbImg);
        } else {
            // Indicate that sb_reference does not exist
            const noSbRef = document.createElement('div');
            noSbRef.className = 'caption';
            noSbRef.textContent = 'No corresponding sb_reference found.';
            wrapper.appendChild(noSbRef);
        }

        // **Append Wrapper to reference-poses-container**
        referenceImageContainer.appendChild(wrapper);
    }

    // **Remove Reference Pose and Corresponding sb_reference Image**
    function removeReferencePose(png) {
        const wrapper = document.getElementById(`pose-wrapper-${png}`);
        if (wrapper) {
            wrapper.remove();
        }
    }

    // **Update the displayed frame using preloaded images**
    function updateFrame() {
        if (preloadedImages.length > 0 && currentFrameIndex >= 0 && currentFrameIndex < preloadedImages.length) {
            const img = preloadedImages[currentFrameIndex];
            if (img) {
                gifImage.src = img.src;
                // Extract frame name from URL
                const urlParts = img.src.split('/');
                currentFrameName = urlParts[urlParts.length - 1];
                // Update frame info
                frameInfo.textContent = `Frame: ${currentFrameIndex + 1} / ${preloadedImages.length}`;
                // Fetch and display notes for the current frame
                fetchAndDisplayNotes(currentGifName, currentFrameName);
            } else {
                // If frame not preloaded, display a placeholder or leave blank
                gifImage.src = '';
                frameInfo.textContent = `Frame: ${currentFrameIndex + 1} / ${preloadedImages.length}`;
                // Clear notes section
                clearNotesSection();
            }
            // Preload next batch if nearing the end of current preloaded frames
            if (currentFrameIndex + preloadBatchSize >= preloadedImages.length && currentFrameIndex + preloadBatchSize < currentFrames.length) {
                preloadBatch(currentFrameIndex + preloadBatchSize);
            }
        }
    }

    // **Update the state of the buttons**
    function updateControls() {
        if (preloadedImages.length > 0) {
            const isSingleFrame = preloadedImages.length === 1;
            prevBtn.disabled = isPlaying || isSingleFrame || currentFrameIndex === 0;
            nextBtn.disabled = isPlaying || isSingleFrame || currentFrameIndex === preloadedImages.length - 1;
            togglePlayBtn.disabled = isSingleFrame; // Disable toggle if only one frame
            togglePlayBtn.textContent = isPlaying ? 'Stop GIF' : 'Play GIF'; // Update button text
        } else {
            prevBtn.disabled = true;
            nextBtn.disabled = true;
            togglePlayBtn.disabled = true;
            togglePlayBtn.textContent = 'Play GIF';
        }
    }

    // **Handle Previous Frame button**
    prevBtn.addEventListener('click', () => {
        if (currentFrameIndex > 0) {
            currentFrameIndex--;
            updateFrame();
            updateControls();
            if (isPlaying) {
                resetPlayInterval();
            }
        }
    });

    // **Handle Next Frame button**
    nextBtn.addEventListener('click', () => {
        if (currentFrameIndex < preloadedImages.length - 1) {
            currentFrameIndex++;
            updateFrame();
            updateControls();
            if (isPlaying) {
                resetPlayInterval();
            }
        }
    });

    // **Handle Keyboard Navigation**
    document.addEventListener('keydown', (event) => {
        // Ensure that the event target is not an input or textarea to avoid interfering with typing
        const tag = event.target.tagName.toLowerCase();
        if (tag === 'input' || tag === 'textarea') {
            return;
        }

        switch(event.key) {
            case 'ArrowLeft':
                // Prevent default behavior like scrolling
                event.preventDefault();
                if (currentFrameIndex > 0) {
                    currentFrameIndex--;
                    updateFrame();
                    updateControls();
                    if (isPlaying) {
                        resetPlayInterval();
                    }
                }
                break;
            case 'ArrowRight':
                // Prevent default behavior like scrolling
                event.preventDefault();
                if (currentFrameIndex < preloadedImages.length - 1) {
                    currentFrameIndex++;
                    updateFrame();
                    updateControls();
                    if (isPlaying) {
                        resetPlayInterval();
                    }
                }
                break;
            case ' ':
                // Toggle Play/Pause with Spacebar
                event.preventDefault();
                togglePlayPause();
                break;
            default:
                // Do nothing for other keys
                break;
        }
    });

    // **Toggle Play/Pause Functionality**
    togglePlayBtn.addEventListener('click', togglePlayPause); // Attach event listener

    function togglePlayPause() {
        if (isPlaying) {
            stopPlayback();
            showNotification('Playback stopped.', false);
            console.log('Playback stopped.');
        } else {
            if (preloadedImages.length === 0) return;
            startPlayback();
            showNotification('Playback started.', false);
            console.log('Playback started.');
        }
        updateControls();
    }

    // **Start Automatic Playback**
    function startPlayback() {
        isPlaying = true;
        togglePlayBtn.textContent = 'Stop GIF';
        togglePlayBtn.classList.add('playing');
        togglePlayBtn.classList.remove('paused');
        playInterval = setInterval(() => {
            currentFrameIndex++;
            if (currentFrameIndex >= preloadedImages.length) {
                currentFrameIndex = 0; // Loop back to the first frame
            }
            updateFrame();
            updateControls();
        }, frameRate);
    }

    // **Stop Automatic Playback**
    function stopPlayback() {
        isPlaying = false;
        togglePlayBtn.textContent = 'Play GIF';
        togglePlayBtn.classList.add('paused');
        togglePlayBtn.classList.remove('playing');
        if (playInterval) {
            clearInterval(playInterval);
            playInterval = null;
        }
    }

    // **Reset Play Interval when manually navigating**
    function resetPlayInterval() {
        if (isPlaying) {
            clearInterval(playInterval);
            playInterval = setInterval(() => {
                currentFrameIndex++;
                if (currentFrameIndex >= preloadedImages.length) {
                    currentFrameIndex = 0; // Loop back to the first frame
                }
                updateFrame();
                updateControls();
            }, frameRate);
        }
    }

    // **Handle Frame Rate Changes**
    frameRateInput.addEventListener('change', () => {
        const newRate = parseInt(frameRateInput.value, 10);
        if (!isNaN(newRate) && newRate >= 10 && newRate <= 1000) {
            frameRate = newRate;
            if (isPlaying) {
                clearInterval(playInterval);
                playInterval = setInterval(() => {
                    currentFrameIndex++;
                    if (currentFrameIndex >= preloadedImages.length) {
                        currentFrameIndex = 0; // Loop back to the first frame
                    }
                    updateFrame();
                    updateControls();
                }, frameRate);
            }
        } else {
            // Reset to previous frame rate if invalid input
            frameRateInput.value = frameRate;
            showNotification('Invalid frame rate. Reset to previous value.', true);
        }
    });

    // **Function to clear all selected poses**
    function clearAllSelectedPoses() {
        const checkedCheckboxes = poseSelectContainer.querySelectorAll('input[type="checkbox"]:checked');
        checkedCheckboxes.forEach(checkbox => {
            checkbox.checked = false;
            removeReferencePose(checkbox.value);
        });
        updateClearAllButtonState(); // Update button state after clearing
    }

    // **Function to update Clear All button state**
    function updateClearAllButtonState() {
        const anyChecked = poseSelectContainer.querySelectorAll('input[type="checkbox"]:checked').length > 0;
        clearAllBtn.disabled = !anyChecked;
    }

    // **Initialize Clear All Button State**
    updateClearAllButtonState();

    // **Event Listener for Clear All Button**
    clearAllBtn.addEventListener('click', () => {
        clearAllSelectedPoses();
    });

    // **Display Reference Pose and Corresponding sb_reference Image**
    function displayReferencePose(png) {
        // Check if the image already exists to prevent duplicates
        if (document.getElementById(`pose-wrapper-${png}`)) {
            return;
        }

        // Create a wrapper div
        const wrapper = document.createElement('div');
        wrapper.className = 'reference-image-wrapper';
        wrapper.id = `pose-wrapper-${png}`;

        // **Extract Base Name for sb_references**
        const baseName = png.replace(/^final_/, '').replace(/\.png$/i, '');
        const sbReferenceName = `${baseName}.jpg`;

        // **Check if sb_reference exists**
        const sbRefExists = sbReferenceFiles.includes(sbReferenceName);

        // **Create Caption for Pose PNG**
        const poseCaption = document.createElement('div');
        poseCaption.className = 'caption';
        poseCaption.textContent = png;

        // **Create Pose Image Element**
        const poseImg = document.createElement('img');
        poseImg.src = `/reference_poses/${png}`;
        poseImg.alt = png;

        poseImg.onload = () => {
            // Image loaded successfully
        };

        poseImg.onerror = () => {
            console.error(`Failed to load Reference Pose: ${png}`);
            poseImg.alt = 'Failed to load image.';
            // Optionally, you can display a placeholder image here
        };

        // **Append Pose Caption and Image to Wrapper**
        wrapper.appendChild(poseCaption);
        wrapper.appendChild(poseImg);

        // **If sb_reference exists, append its Caption and Image**
        if (sbRefExists) {
            const sbCaption = document.createElement('div');
            sbCaption.className = 'caption';
            sbCaption.textContent = sbReferenceName;

            const sbImg = document.createElement('img');
            sbImg.src = `/sb_references/${sbReferenceName}`;
            sbImg.alt = sbReferenceName;

            sbImg.onload = () => {
                // Image loaded successfully
            };

            sbImg.onerror = () => {
                console.error(`Failed to load sb_reference Image: ${sbReferenceName}`);
                sbImg.alt = 'Failed to load image.';
                // Optionally, you can display a placeholder image here
            };

            wrapper.appendChild(sbCaption);
            wrapper.appendChild(sbImg);
        } else {
            // Indicate that sb_reference does not exist
            const noSbRef = document.createElement('div');
            noSbRef.className = 'caption';
            noSbRef.textContent = 'No corresponding sb_reference found.';
            wrapper.appendChild(noSbRef);
        }

        // **Append Wrapper to reference-poses-container**
        referenceImageContainer.appendChild(wrapper);
    }

    // **Remove Reference Pose and Corresponding sb_reference Image**
    function removeReferencePose(png) {
        const wrapper = document.getElementById(`pose-wrapper-${png}`);
        if (wrapper) {
            wrapper.remove();
        }
    }

    // **Function to Fetch and Display Notes for the Current Frame**
    function fetchAndDisplayNotes(gifName, frameName) {
        if (!gifName || !frameName) {
            clearNotesSection();
            return;
        }

        fetch(`/api/gifs/${encodeURIComponent(gifName)}/frames/${encodeURIComponent(frameName)}/notes`, { cache: 'no-store' })
            .then(response => response.json())
            .then(notes => {
                displayNotes(notes);
            })
            .catch(error => {
                console.error('Error fetching notes:', error);
                // Optionally, display an error message to the user
                showNotification('Failed to load notes for this frame.', true);
            });
    }

    // **Function to Display Notes in the UI**
    function displayNotes(notes) {
        // Clear existing notes and "No notes" messages
        const existingNotes = notesList.querySelectorAll('.note-item, .no-notes');
        existingNotes.forEach(note => note.remove());

        if (notes.length === 0) {
            // Add "No notes" message only once
            const noNotes = document.createElement('p');
            noNotes.className = 'no-notes';
            noNotes.textContent = 'No notes for this frame.';
            notesList.appendChild(noNotes);
            return;
        }

        // Create note elements
        notes.forEach(note => {
            const noteDiv = document.createElement('div');
            noteDiv.className = 'note-item';
            noteDiv.dataset.noteId = note.id; // Store note ID for deletion

            const timestampDiv = document.createElement('div');
            timestampDiv.className = 'note-timestamp';
            const date = new Date(note.timestamp);
            timestampDiv.textContent = date.toLocaleString();

            const contentDiv = document.createElement('div');
            contentDiv.className = 'note-content';
            contentDiv.textContent = note.content;

            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = 'Delete';

            // **Event Listener for Delete Button**
            deleteBtn.addEventListener('click', () => {
                deleteNote(note.id);
            });

            noteDiv.appendChild(timestampDiv);
            noteDiv.appendChild(contentDiv);
            noteDiv.appendChild(deleteBtn);

            notesList.appendChild(noteDiv);
        });
    }

    // **Function to Delete a Note**
    function deleteNote(noteId) {
        if (!currentGifName || !currentFrameName) {
            showNotification('No frame selected.', true);
            return;
        }

        // Confirm deletion
        const confirmDelete = confirm('Are you sure you want to delete this note?');
        if (!confirmDelete) return;

        // Send DELETE request to the server
        fetch(`/api/gifs/${encodeURIComponent(currentGifName)}/frames/${encodeURIComponent(currentFrameName)}/notes/${encodeURIComponent(noteId)}`, {
            method: 'DELETE'
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw err; });
                }
                return response.json();
            })
            .then(data => {
                console.log(data.message);
                // Show success notification
                showNotification('Note deleted successfully.', false);
                // Refresh the notes list
                fetchAndDisplayNotes(currentGifName, currentFrameName);
                // Also refresh the All Notes tab if it's active
                const activeTab = document.querySelector('.tabcontent.active');
                if (activeTab.id === 'AllNotes') {
                    fetchAndDisplayAllNotes();
                }
            })
            .catch(error => {
                console.error('Error deleting note:', error);
                // Display a user-friendly error message
                if (error.error) {
                    showNotification(`Error: ${error.error}`, true);
                } else {
                    showNotification('An unexpected error occurred while deleting the note. Please try again.', true);
                }
            });
    }

    // **Function to Clear Notes Section**
    function clearNotesSection() {
        // Clear existing notes and "No notes" messages
        const existingNotes = notesList.querySelectorAll('.note-item, .no-notes');
        existingNotes.forEach(note => note.remove());

        // Optionally, add a placeholder message
        const placeholder = document.createElement('p');
        placeholder.className = 'no-notes';
        placeholder.textContent = 'No GIF selected.';
        notesList.appendChild(placeholder);
    }

    // **Event Listener for Save Note Button**
    saveNoteBtn.addEventListener('click', () => {
        const noteContent = noteInput.value.trim();

        if (!noteContent) {
            showNotification('Please enter a note before saving.', true);
            return;
        }

        if (!currentGifName || !currentFrameName) {
            showNotification('No frame selected to attach the note.', true);
            return;
        }

        // Prepare the data to send
        const data = {
            content: noteContent
        };

        // Send POST request to save the note
        fetch(`/api/gifs/${encodeURIComponent(currentGifName)}/frames/${encodeURIComponent(currentFrameName)}/notes`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw err; });
                }
                return response.json();
            })
            .then(newNote => {
                console.log('Note saved:', newNote);
                // Clear the input box
                noteInput.value = '';
                // Show success notification
                showNotification('Note saved successfully.', false);
                // Fetch and display updated notes
                fetchAndDisplayNotes(currentGifName, currentFrameName);
                // Also refresh the All Notes tab if it's active
                const activeTab = document.querySelector('.tabcontent.active');
                if (activeTab.id === 'AllNotes') {
                    fetchAndDisplayAllNotes();
                }
            })
            .catch(error => {
                console.error('Error saving note:', error);
                // Display a user-friendly error message
                if (error.error) {
                    showNotification(`Error: ${error.error}`, true);
                } else {
                    showNotification('An unexpected error occurred while saving the note. Please try again.', true);
                }
            });
    });

    // **Function to Show Notifications**
    function showNotification(message, isError = false) {
        notification.textContent = message;
        notification.style.backgroundColor = isError ? '#f44336' : '#4CAF50'; // Red for errors, green for success
        notification.style.display = 'block';
        notification.classList.add('show');

        // Hide after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            notification.classList.add('hide');
            // After transition ends, hide the notification
            setTimeout(() => {
                notification.style.display = 'none';
                notification.classList.remove('hide');
            }, 500);
        }, 3000);
    }

    // **Function to Fetch and Display All Notes**
    function fetchAndDisplayAllNotes() {
        console.log('fetchAndDisplayAllNotes called.');
        updateAllNotesBtn.disabled = true; // Disable the button
        const originalBtnText = updateAllNotesBtn.textContent;
        updateAllNotesBtn.textContent = 'Updating...'; // Change button text
        allNotesList.innerHTML = '<p>Loading all notes...</p>'; // Show loading message
        fetch('/api/notes', { cache: 'no-store' }) // Fetch all notes
            .then(response => {
                console.log('fetchAndDisplayAllNotes: Received response status', response.status);
                if (!response.ok) {
                    throw new Error('Failed to fetch all notes.');
                }
                return response.json();
            })
            .then(allNotes => {
                console.log('All Notes fetched:', allNotes); // Debug log
                displayAllNotes(allNotes);
                updateAllNotesBtn.textContent = originalBtnText; // Restore original text
                updateAllNotesBtn.disabled = false; // Re-enable the button
                showNotification('Notes updated successfully.', false); // Success notification
            })
            .catch(error => {
                console.error('Error fetching all notes:', error);
                allNotesList.innerHTML = '<p>Error loading all notes.</p>';
                showNotification('Failed to load all notes.', true);
                updateAllNotesBtn.textContent = originalBtnText; // Restore original text
                updateAllNotesBtn.disabled = false; // Re-enable the button
            });
    }

    // **Function to Display All Notes in the UI**
    function displayAllNotes(allNotes) {
        console.log('displayAllNotes called with:', allNotes); // Debug log

        // Clear existing content
        allNotesList.innerHTML = '';

        if (!allNotes || Object.keys(allNotes).length === 0) {
            allNotesList.innerHTML = '<p>No notes available.</p>';
            console.log('displayAllNotes: No notes available.');
            showNotification('No notes available.', false);
            return;
        }

        // Iterate over each GIF
        for (const [gifName, frames] of Object.entries(allNotes)) {
            console.log(`Processing GIF: ${gifName}`); // Debug log
            // Create GIF container
            const gifContainer = document.createElement('div');
            gifContainer.className = 'gif-container';
            gifContainer.style.marginBottom = '20px';

            const gifHeader = document.createElement('h3');
            gifHeader.textContent = `GIF: ${gifName}`;
            gifContainer.appendChild(gifHeader);

            // Iterate over each frame
            for (const [frameName, notes] of Object.entries(frames)) {
                if (notes.length === 0) continue; // Skip frames with no notes

                console.log(`Processing Frame: ${frameName} with ${notes.length} notes`); // Debug log

                // Create Frame container
                const frameContainer = document.createElement('div');
                frameContainer.className = 'frame-container';
                frameContainer.style.marginLeft = '20px';
                frameContainer.style.marginBottom = '10px';

                const frameHeader = document.createElement('h4');
                frameHeader.textContent = `Frame: ${frameName}`;
                frameContainer.appendChild(frameHeader);

                // Create list of notes
                const notesListElement = document.createElement('ul');
                notesListElement.style.listStyleType = 'disc';
                notesListElement.style.marginLeft = '20px';

                notes.forEach(note => {
                    const noteItem = document.createElement('li');
                    noteItem.textContent = note.content + ` (Added on ${new Date(note.timestamp).toLocaleString()})`;
                    notesListElement.appendChild(noteItem);
                });

                frameContainer.appendChild(notesListElement);
                gifContainer.appendChild(frameContainer);
            }

            allNotesList.appendChild(gifContainer);
        }

        // If no notes are present across all GIFs
        if (allNotesList.innerHTML.trim() === '') {
            allNotesList.innerHTML = '<p>No notes available.</p>';
            console.log('displayAllNotes: No notes available after processing.');
        }
    }

    // **Add Event Listeners for Tabs**
    // Assuming you have buttons with class 'tablinks' and data attributes to identify tabs
    const tabButtons = document.querySelectorAll('.tablinks');

    console.log('Tab Buttons found:', tabButtons.length); // Debug log

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.getAttribute('data-tab');

            // Remove 'active' class from all buttons and tab contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tabcontent').forEach(tab => tab.classList.remove('active'));

            // Add 'active' class to the clicked button and corresponding tab content
            button.classList.add('active');
            document.getElementById(tabName).classList.add('active');

            // If 'AllNotes' tab is activated, fetch and display all notes
            if (tabName === 'AllNotes') {
                fetchAndDisplayAllNotes();
            }
        });
    });

    // **Attach Event Listener for "Update All Notes" Button**
    updateAllNotesBtn.addEventListener('click', () => {
        console.log('Update All Notes button clicked.');
        fetchAndDisplayAllNotes();
    });

    // **Initial Fetch Calls**
    fetchGifs();
    fetchReferencePoses();
    fetchSbReferences();
});
