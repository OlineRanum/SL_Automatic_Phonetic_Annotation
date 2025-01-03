/**************************************
 * GLOBAL (MO-CAP) FUNCTION DEFINED FIRST
 * so we can call it inline in HTML or 
 * from anywhere in the code.
 **************************************/
function fetchMoCapGifs() {
    fetch('/api/mocap_gifs', { cache: 'no-store' })
        .then(response => response.json())
        .then(gifs => {
            const mocapGifSelect = document.getElementById('mocap-gif-select');
            if (gifs.length === 0) {
                mocapGifSelect.innerHTML = '<option value="">No GIFs available</option>';
                return;
            }
            mocapGifSelect.innerHTML = '<option value="">--Select a GIF--</option>';
            gifs.forEach(gif => {
                const option = document.createElement('option');
                option.value = gif;
                option.textContent = gif;
                mocapGifSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error fetching MoCap GIFs:', error);
            document.getElementById('mocap-gif-select').innerHTML = '<option value="">Error loading GIFs</option>';
        });
}



/**************************************
 * MAIN SCRIPT (DOMContentLoaded)
 **************************************/
document.addEventListener('DOMContentLoaded', () => {
    "use strict";
    
    /**************************************
     * 1) COMMON ELEMENTS + VARIABLES
     **************************************/
    
    // -------- Data Viewer Elements --------
    const gifSelect          = document.getElementById('gif-select');
    const gifImage           = document.getElementById('gif-image');
    const prevBtn            = document.getElementById('prev-btn');
    const nextBtn            = document.getElementById('next-btn');
    const togglePlayBtn      = document.getElementById('toggle-play-btn');
    const frameInfo          = document.getElementById('frame-info');
    const frameRateInput     = document.getElementById('frame-rate');
    const loadingIndicator   = document.getElementById('loading');
    const progressBar        = document.getElementById('progress-bar');

    // -------- Reference Poses Viewer Elements --------
    const poseSelectContainer    = document.getElementById('pose-select-container');
    const referenceImageContainer= document.getElementById('reference-poses-container');
    const poseLoading            = document.getElementById('pose-loading');
    const clearAllBtn            = document.getElementById('clear-all-btn');

    // -------- Note-Taking Elements (Data Viewer Only) --------
    const noteInput      = document.getElementById('note-input');
    const saveNoteBtn    = document.getElementById('save-note-btn');
    const notesList      = document.getElementById('notes-list');

    // -------- All Notes Elements (Data Viewer Only) --------
    const allNotesList       = document.getElementById('all-notes-list');
    const updateAllNotesBtn  = document.getElementById('update-all-notes-btn');

    // -------- Notification Element (for both) --------
    const notification = document.getElementById('notification');


    /**************************************
     * 2) DATA VIEWER VARIABLES (WITH NOTES)
     **************************************/
    let currentGifName     = '';
    let currentFrameName   = '';
    let currentFrames      = [];
    let preloadedImages    = [];
    let currentFrameIndex  = 0;
    let isPlaying          = false;
    let playInterval       = null;
    let frameRate          = 200;    // 200ms per frame (5 fps)
    const preloadBatchSize = 1000;    // Number of frames to preload at a time

    // Reference Poses / sb_references
    let referencePoses     = [];
    let sbReferenceFiles   = [];

    /**************************************
     * 3) MO-CAP TAB VARIABLES
     **************************************/
    let mocapCurrentGifName     = '';
    let mocapCurrentFrames      = [];
    let mocapPreloadedImages    = [];
    let mocapCurrentFrameIndex  = 0;
    let mocapIsPlaying          = false;
    let mocapPlayInterval       = null;
    let mocapFrameRate          = 200;  // also 200ms
    const mocapPreloadBatchSize = 2000;

    const mocapFrameInput      = document.getElementById('mocap-frame-input');
    const mocapSaveFramesBtn   = document.getElementById('mocap-save-frames-btn');
    const mocapSavedFramesList = document.getElementById('mocap-saved-frames-list');
    const mocapRemoveInput    = document.getElementById('mocap-remove-input');
    const mocapRemoveFramesBtn= document.getElementById('mocap-remove-frames-btn');



    /**************************************
     * 4) DATA VIEWER LOGIC
     **************************************/

    // ---------- (A) FETCH & POPULATE GIF DROPDOWN ----------
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

    // ---------- (B) WHEN USER SELECTS A GIF ----------
    gifSelect.addEventListener('change', () => {
        const selectedGif = gifSelect.value;
        if (selectedGif) {
            currentGifName = selectedGif; 
            // Reset playback
            stopPlayback();

            // Show loading indicator
            loadingIndicator.style.display = 'block';
            loadingIndicator.innerHTML = '<span>Loading frames...</span><div id="progress-container"><div id="progress-bar"></div></div>';
            progressBar.style.width = '0%';

            // Fetch frames
            fetch(`/api/gifs/${encodeURIComponent(selectedGif)}/frames`, { cache: 'no-store' })
                .then(response => {
                    if (!response.ok) throw new Error('Frames not found.');
                    return response.json();
                })
                .then(frames => {
                    if (frames.length === 0) throw new Error('No frames available for this GIF.');

                    currentFrames = frames;
                    currentFrameIndex = 0;
                    preloadedImages = new Array(currentFrames.length).fill(null);
                    preloadBatch(currentFrameIndex); // Start batch preloading
                })
                .catch(error => {
                    console.error('Error fetching frames:', error);
                    loadingIndicator.innerHTML = 'Error loading frames.';
                });
        } else {
            // Reset if no GIF is selected
            currentGifName    = '';
            currentFrames     = [];
            preloadedImages   = [];
            currentFrameIndex = 0;
            gifImage.src      = '';
            frameInfo.textContent = 'Frame: 0';
            updateControls();
            togglePlayBtn.disabled = true;
            loadingIndicator.style.display = 'none';
            progressBar.style.width        = '0%';
            // Clear notes
            clearNotesSection();
        }
    });

    // ---------- (C) PRELOAD BATCH FOR DATA VIEWER ----------
    function preloadBatch(startIndex) {
        const endIndex   = Math.min(startIndex + preloadBatchSize, currentFrames.length);
        const promises   = [];
        let loadedCount  = 0;
        const totalToLoad= endIndex - startIndex;

        for (let i = startIndex; i < endIndex; i++) {
            if (!preloadedImages[i]) {
                const img = new Image();
                img.src    = currentFrames[i];
                const promise = new Promise((resolve) => {
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
                        resolve(); // continue
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
                // Preload next batch
                if (endIndex < currentFrames.length) {
                    preloadBatch(endIndex);
                }
                // Hide loading if all loaded
                if (preloadedImages.every(img => img !== null)) {
                    loadingIndicator.style.display = 'none';
                    updateProgress(0);
                }
            })
            .catch(err => {
                console.error('Error preloading frames:', err);
                loadingIndicator.innerHTML = 'Error loading frames.';
                updateProgress(0);
            });
    }

    // ---------- (D) UPDATE FRAME (DATA VIEWER) ----------
    function updateFrame() {
        if (preloadedImages.length > 0 && currentFrameIndex >= 0 && currentFrameIndex < preloadedImages.length) {
            const imgObj = preloadedImages[currentFrameIndex];
            if (imgObj) {
                gifImage.src = imgObj.src;
                const urlParts = imgObj.src.split('/');
                currentFrameName = urlParts[urlParts.length - 1];
                frameInfo.textContent = `Frame: ${currentFrameIndex + 1} / ${preloadedImages.length}`;
                // Fetch notes for the current frame
                fetchAndDisplayNotes(currentGifName, currentFrameName);
            } else {
                // If not preloaded
                gifImage.src = '';
                frameInfo.textContent = `Frame: ${currentFrameIndex + 1} / ${preloadedImages.length}`;
                clearNotesSection();
            }
            // Preload the next batch if nearing the end
            if ((currentFrameIndex + preloadBatchSize) >= preloadedImages.length 
                && (currentFrameIndex + preloadBatchSize) < currentFrames.length) {
                preloadBatch(currentFrameIndex + preloadBatchSize);
            }
        }
        }

    // ---------- (E) PROGRESS BAR UPDATE ----------
    function updateProgress(percent) {
        progressBar.style.width = `${percent}%`;
    }

    // ---------- (F) CONTROLS (DATA VIEWER) ----------
    function updateControls() {
        if (preloadedImages.length > 0) {
            const isSingleFrame = (preloadedImages.length === 1);
            prevBtn.disabled       = isPlaying || isSingleFrame || currentFrameIndex === 0;
            nextBtn.disabled       = isPlaying || isSingleFrame || currentFrameIndex === preloadedImages.length - 1;
            togglePlayBtn.disabled = isSingleFrame;
            togglePlayBtn.textContent = isPlaying ? 'Stop GIF' : 'Play GIF';
        } else {
            prevBtn.disabled       = true;
            nextBtn.disabled       = true;
            togglePlayBtn.disabled = true;
            togglePlayBtn.textContent = 'Play GIF';
        }
    }

    // ---------- (G) PLAYBACK (DATA VIEWER) ----------
    function startPlayback() {
        isPlaying = true;
        togglePlayBtn.textContent = 'Stop GIF';
        togglePlayBtn.classList.add('playing');
        togglePlayBtn.classList.remove('paused');
        playInterval = setInterval(() => {
            currentFrameIndex++;
            if (currentFrameIndex >= preloadedImages.length) {
                currentFrameIndex = 0;
            }
            updateFrame();
            updateControls();
        }, frameRate);
    }
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
    function resetPlayInterval() {
        if (isPlaying) {
            clearInterval(playInterval);
            playInterval = setInterval(() => {
                currentFrameIndex++;
                if (currentFrameIndex >= preloadedImages.length) {
                    currentFrameIndex = 0;
                }
                updateFrame();
                updateControls();
            }, frameRate);
        }
    }
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

    // ---------- (H) DATA VIEWER BUTTON LISTENERS ----------
    prevBtn.addEventListener('click', () => {
        if (currentFrameIndex > 0) {
            currentFrameIndex--;
            updateFrame();
            updateControls();
            if (isPlaying) resetPlayInterval();
        }
    });
    nextBtn.addEventListener('click', () => {
        if (currentFrameIndex < preloadedImages.length - 1) {
            currentFrameIndex++;
            updateFrame();
            updateControls();
            if (isPlaying) resetPlayInterval();
        }
    });
    togglePlayBtn.addEventListener('click', togglePlayPause);

    // ---------- (I) FRAME RATE CHANGE ----------
    frameRateInput.addEventListener('change', () => {
        const newRate = parseInt(frameRateInput.value, 10);
        if (!isNaN(newRate) && newRate >= 10 && newRate <= 1000) {
            frameRate = newRate;
            if (isPlaying) {
                clearInterval(playInterval);
                playInterval = setInterval(() => {
                    currentFrameIndex++;
                    if (currentFrameIndex >= preloadedImages.length) {
                        currentFrameIndex = 0;
                    }
                    updateFrame();
                    updateControls();
                }, frameRate);
            }
        } else {
            frameRateInput.value = frameRate;
            showNotification('Invalid frame rate. Reset to previous value.', true);
        }
    });

    /**************************************
     * 5) MO-CAP LOGIC (NEARLY IDENTICAL)
     **************************************/

    // ---------- (A) SELECTING A MoCap GIF ----------
    const mocapGifSelect        = document.getElementById('mocap-gif-select');
    const mocapLoadingIndicator = document.getElementById('mocap-loading');
    const mocapProgressBar      = document.getElementById('mocap-progress-bar');
    const mocapGifImage         = document.getElementById('mocap-gif-image');
    const mocapFrameInfo        = document.getElementById('mocap-frame-info');
    const mocapPrevBtn          = document.getElementById('mocap-prev-btn');
    const mocapNextBtn          = document.getElementById('mocap-next-btn');
    const mocapTogglePlayBtn    = document.getElementById('mocap-toggle-play-btn');
    const mocapFrameRateInput   = document.getElementById('mocap-frame-rate');

    // When user selects a MoCap GIF
    mocapGifSelect.addEventListener('change', () => {
        const selectedGif = mocapGifSelect.value;
        if (selectedGif) {
            // 1) Set the current GIF name
            mocapCurrentGifName = selectedGif;
    
            // 2) Stop any current playback for safety
            mocapStopPlayback();
    
            // 3) Show "Loading..." (or reset UI)
            mocapLoadingIndicator.style.display = 'block';
            mocapProgressBar.style.width = '0%';
    
            // 4) Fetch actual frames for playback
            fetch(`/api/mocap_gifs/${encodeURIComponent(selectedGif)}/frames`, { cache: 'no-store' })
                .then(response => {
                    if (!response.ok) throw new Error('Frames not found.');
                    return response.json();
                })
                .then(frames => {
                    if (frames.length === 0) {
                        throw new Error('No frames available for this GIF.');
                    }
    
                    // 4a) Store them in local arrays so the user can watch or iterate
                    mocapCurrentFrames = frames;
                    mocapCurrentFrameIndex = 0;
                    mocapPreloadedImages = new Array(mocapCurrentFrames.length).fill(null);
                    mocapPreloadBatch(mocapCurrentFrameIndex);
                })
                .catch(error => {
                    console.error('Error fetching MoCap frames:', error);
                    mocapLoadingIndicator.innerHTML = 'Error loading frames.';
                });
    
            // 5) **AFTER** or simultaneously, fetch the previously stored frame indexes:
            fetchMoCapSelectedFrames();  
        } else {
            // If user clears selection, reset everything
            mocapCurrentGifName      = '';
            mocapCurrentFrames       = [];
            mocapPreloadedImages     = [];
            mocapCurrentFrameIndex   = 0;
            mocapGifImage.src        = '';
            mocapFrameInfo.textContent = 'Frame: 0';
            mocapLoadingIndicator.style.display = 'none';
            mocapProgressBar.style.width        = '0%';
    
            // Clear the displayed list of saved frames
            mocapSavedFramesList.innerHTML = '<p>No MoCap GIF selected.</p>';
        }
    });

    // ---------- (B) MoCap PRELOAD BATCH ----------
    function mocapPreloadBatch(startIndex) {
        const endIndex  = Math.min(startIndex + mocapPreloadBatchSize, mocapCurrentFrames.length);
        const promises  = [];
        let loadedCount = 0;

        for (let i = startIndex; i < endIndex; i++) {
            console.log(`Loading frame index: ${i}, url=${mocapCurrentFrames[i]}`);
            if (!mocapPreloadedImages[i]) {
                const img = new Image();
                img.src = mocapCurrentFrames[i];
                const promise = new Promise(resolve => {
                    img.onload = () => {
                        console.log(`Frame ${i} loaded successfully`);
                        mocapPreloadedImages[i] = img;
                        loadedCount++;
                        mocapProgressBar.style.width = `${(loadedCount / (endIndex - startIndex)) * 100}%`;
                        resolve();
                    };
                    img.onerror = () => {
                        console.error(`Frame ${i} failed`);
                        console.error(`Failed to load frame ${i + 1}`);
                        loadedCount++;
                        mocapProgressBar.style.width = `${(loadedCount / (endIndex - startIndex)) * 100}%`;
                        resolve();
                    };
                });
                promises.push(promise);
            }
        }

        Promise.all(promises)
            .then(() => {
                console.log(`Batch from ${startIndex} to ${endIndex} is done`);

                mocapUpdateFrame();
                mocapUpdateControls(); 
    
                // You could do additional preloading if you like
                if (endIndex >= mocapCurrentFrames.length) {
                    mocapLoadingIndicator.style.display = 'none';
                } 
            })
            .catch(err => {
                console.error('Error preloading MoCap frames:', err);
                mocapLoadingIndicator.innerHTML = 'Error loading frames.';
            });
    }

    // ---------- (C) MoCap UPDATE FRAME ----------
    function mocapUpdateFrame() {
        if (mocapPreloadedImages[mocapCurrentFrameIndex]) {
            mocapGifImage.src = mocapPreloadedImages[mocapCurrentFrameIndex].src;
            mocapFrameInfo.textContent = `Frame: ${mocapCurrentFrameIndex + 1} / ${mocapPreloadedImages.length}`;
        }
    }

    // ---------- (D) MoCap PLAYBACK ----------
    function mocapStartPlayback() {
        mocapIsPlaying = true;
        mocapTogglePlayBtn.textContent = 'Stop GIF';
        mocapPlayInterval = setInterval(() => {
            mocapCurrentFrameIndex = (mocapCurrentFrameIndex + 1) % mocapPreloadedImages.length;
            mocapUpdateFrame();
        }, mocapFrameRate);
    }
    function mocapStopPlayback() {
        mocapIsPlaying = false;
        clearInterval(mocapPlayInterval);
        mocapTogglePlayBtn.textContent = 'Play GIF';
    }

    // ---------- (E) MoCap BUTTON LISTENERS ----------
    mocapPrevBtn.addEventListener('click', () => {
        if (mocapCurrentFrameIndex > 0) {
            mocapCurrentFrameIndex--;
            mocapUpdateFrame();
            mocapUpdateControls();
        }
    });
    mocapNextBtn.addEventListener('click', () => {
        if (mocapCurrentFrameIndex < mocapPreloadedImages.length - 1) {
            mocapCurrentFrameIndex++;
            mocapUpdateFrame();
            mocapUpdateControls();
        }
    });
    mocapTogglePlayBtn.addEventListener('click', () => {
        if (mocapIsPlaying) {
            mocapStopPlayback();
        } else {
            mocapStartPlayback();
        }
    });

    // ---------- (F) MoCap FRAME RATE ----------
    mocapFrameRateInput.addEventListener('change', () => {
        const newRate = parseInt(mocapFrameRateInput.value, 10);
        if (!isNaN(newRate) && newRate >= 10 && newRate <= 1000) {
            mocapFrameRate = newRate;
            if (mocapIsPlaying) {
                clearInterval(mocapPlayInterval);
                mocapPlayInterval = setInterval(() => {
                    mocapCurrentFrameIndex = (mocapCurrentFrameIndex + 1) % mocapPreloadedImages.length;
                    mocapUpdateFrame();
                }, mocapFrameRate);
            }
        } else {
            mocapFrameRateInput.value = mocapFrameRate;
            showNotification('Invalid MoCap frame rate. Reset to previous value.', true);
        }
    });

    // ---------- (G) MoCap Update controls ----------
    function mocapUpdateControls() {
        // If we have frames loaded:
        if (mocapPreloadedImages.length > 0) {
            const isSingleFrame = (mocapPreloadedImages.length === 1);
    
            // If playing, or if on the first frame, or only one frame total ⇒ disable Prev
            mocapPrevBtn.disabled = (mocapIsPlaying || mocapCurrentFrameIndex === 0 || isSingleFrame);
    
            // If playing, or if on the last frame, or only one frame total ⇒ disable Next
            mocapNextBtn.disabled = (
                mocapIsPlaying ||
                mocapCurrentFrameIndex === (mocapPreloadedImages.length - 1) ||
                isSingleFrame
            );
    
            // Play/Stop button is disabled if only 1 frame
            mocapTogglePlayBtn.disabled = isSingleFrame;
            mocapTogglePlayBtn.textContent = mocapIsPlaying ? 'Stop GIF' : 'Play GIF';
        } else {
            // No frames loaded yet ⇒ disable everything
            mocapPrevBtn.disabled       = true;
            mocapNextBtn.disabled       = true;
            mocapTogglePlayBtn.disabled = true;
            mocapTogglePlayBtn.textContent = 'Play GIF';
        }
    }
    
    // ---------- (H) MoCap SAVE FRAMES ----------
    mocapSaveFramesBtn.addEventListener('click', () => {
        // If no MoCap GIF is selected, do nothing
        if (!mocapCurrentGifName) {
           alert('No MoCap GIF is selected!');
           return;
        }
        const inputValue = mocapFrameInput.value.trim();
        if (!inputValue) {
           alert('Please enter a single index or a range (e.g. 210 or 211-234).');
           return;
        }
        
        // Send a POST request to store these frames
        const baseName = encodeURIComponent(mocapCurrentGifName); // e.g. "myMoCap.gif"
        fetch(`/api/mocap_gifs/${baseName}/selected_frames`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rangeOrIndex: inputValue })
        })
        .then(response => {
            if (!response.ok) throw new Error('Failed to store frames.');
            return response.json();
        })
        .then(data => {
            console.log('Frames stored:', data.frames);
            // Clear the input
            mocapFrameInput.value = '';
            // Refresh the list
            fetchMoCapSelectedFrames();
        })
        .catch(err => {
            console.error('Error storing frames:', err);
            alert('Error storing frames. Check console.');
        });
    });
    

    mocapRemoveFramesBtn.addEventListener('click', () => {
        if (!mocapCurrentGifName) {
          alert('No MoCap GIF is selected!');
          return;
        }
        const value = mocapRemoveInput.value.trim();
        if (!value.includes('-')) {
          alert('Please enter a range, e.g. 211-234.');
          return;
        }
      
        const [startStr, endStr] = value.split('-').map(s => s.trim());
        const start = parseInt(startStr, 10);
        const end   = parseInt(endStr, 10);
        if (isNaN(start) || isNaN(end) || start > end) {
          alert('Invalid range format. Use something like 211-234');
          return;
        }
      
        const encoded = encodeURIComponent(mocapCurrentGifName);
        // Send DELETE with query params
        fetch(`/api/mocap_gifs/${encoded}/selected_frames?start=${start}&end=${end}`, {
          method: 'DELETE'
        })
        .then(response => {
          if (!response.ok) throw new Error('Failed to remove frames.');
          return response.json();
        })
        .then(data => {
          console.log('Removed frames in range:', data);
          mocapRemoveInput.value = '';
          // Refresh the displayed frames
          fetchMoCapSelectedFrames();
        })
        .catch(err => {
          console.error('Error removing frames:', err);
          alert('Error removing frames. Check console.');
        });
      });

      function fetchMoCapSelectedFrames() {
        if (!mocapCurrentGifName) {
            // If no MoCap GIF is selected, clear the list
            mocapSavedFramesList.innerHTML = '<p>No MoCap GIF selected.</p>';
            return;
        }
    
        const baseName = encodeURIComponent(mocapCurrentGifName);
        fetch(`/api/mocap_gifs/${baseName}/selected_frames`)
            .then(response => response.json())
            .then(indexes => {
                const ranges = formatIndicesAsRanges(indexes); // Format indexes into ranges
                displayMoCapSelectedFrames(ranges);
            })
            .catch(error => {
                console.error('Error fetching MoCap selected frames:', error);
                mocapSavedFramesList.innerHTML = '<p>Error loading selected frames.</p>';
            });
    }
    
    
    function displayMoCapSelectedFrames(ranges) {
        if (!ranges || ranges.length === 0) {
            mocapSavedFramesList.innerHTML = '<p>No frames stored.</p>';
            return;
        }
    
        // Clear the existing list
        mocapSavedFramesList.innerHTML = '';
    
        // Display each range as a list item or chip
        ranges.forEach(range => {
            const rangeItem = document.createElement('div');
            rangeItem.textContent = range;
            rangeItem.style.padding = '5px 10px';
            rangeItem.style.margin = '5px 0';
            rangeItem.style.backgroundColor = '#f0f0f0';
            rangeItem.style.border = '1px solid #ccc';
            rangeItem.style.borderRadius = '5px';
    
            // Optionally add a delete button for the range
            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = 'Delete';
            deleteBtn.style.marginLeft = '10px';
            deleteBtn.addEventListener('click', () => {
                deleteRange(range); // Implement delete logic for ranges
            });
    
            rangeItem.appendChild(deleteBtn);
            mocapSavedFramesList.appendChild(rangeItem);
        });
    }
    
    
    function deleteMoCapFrameIndex(idx) {
        if (!mocapCurrentGifName) return;
        const baseName = encodeURIComponent(mocapCurrentGifName);
        fetch(`/api/mocap_gifs/${baseName}/selected_frames/${idx}`, {
            method: 'DELETE'
        })
        .then(response => {
            if (!response.ok) throw new Error('Failed to delete frame index.');
            return response.json();
        })
        .then(data => {
            console.log(`Index ${idx} removed:`, data.frames);
            // Refresh the list
            fetchMoCapSelectedFrames();
        })
        .catch(error => {
            console.error('Error deleting MoCap index:', error);
            alert('Error deleting index. Check console.');
        });
    }

    function formatIndicesAsRanges(indices) {
        if (!indices.length) return [];
        
        // Sort indices in ascending order
        indices.sort((a, b) => a - b);
    
        const ranges = [];
        let start = indices[0];
        let end = start;
    
        for (let i = 1; i < indices.length; i++) {
            if (indices[i] === end + 1) {
                // Extend the current range
                end = indices[i];
            } else {
                // Close the current range and start a new one
                ranges.push(start === end ? `${start}` : `${start}-${end}`);
                start = indices[i];
                end = start;
            }
        }
    
        // Add the final range
        ranges.push(start === end ? `${start}` : `${start}-${end}`);
        return ranges;
    }

    function updateDisplayedIndices() {
        const ranges = formatIndicesAsRanges(Object.keys(frameIndices).map(Number));
        indexListContainer.innerHTML = ''; // Clear existing content
    
        if (ranges.length === 0) {
            indexListContainer.textContent = 'No indices stored.';
            return;
        }
    
        ranges.forEach(range => {
            const rangeDiv = document.createElement('div');
            rangeDiv.className = 'range-item';
            rangeDiv.textContent = range;
    
            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = 'Delete Range';
            deleteBtn.addEventListener('click', () => {
                deleteRange(range);
            });
    
            rangeDiv.appendChild(deleteBtn);
            indexListContainer.appendChild(rangeDiv);
        });
    }

    function deleteRange(range) {
        const [start, end] = range.split('-').map(Number);
        if (end === undefined) {
            // Single index
            delete frameIndices[start];
        } else {
            // Range of indices
            for (let i = start; i <= end; i++) {
                delete frameIndices[i];
            }
        }
        updateDisplayedIndices();
    }
    
    
    
    
    /**************************************
     * 6) REFERENCE POSES, CLEAR-ALL, ETC.
     **************************************/
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
                    checkbox.type  = 'checkbox';
                    checkbox.id    = `pose-${png}`;
                    checkbox.value = png;
                    checkbox.setAttribute('aria-label', `Select ${png} Pose`);

                    const label = document.createElement('label');
                    label.htmlFor = `pose-${png}`;
                    label.textContent = png;

                    // Thumbnail
                    const thumbnail = document.createElement('img');
                    thumbnail.src   = `/reference_poses/${png}`;
                    thumbnail.alt   = `${png} thumbnail`;
                    thumbnail.style.width = '50px';
                    thumbnail.style.height= 'auto';
                    thumbnail.style.marginRight = '10px';
                    thumbnail.style.border = '1px solid #ccc';
                    thumbnail.style.objectFit = 'cover';
                    thumbnail.style.flexShrink = '0';

                    const container = document.createElement('div');
                    container.style.display = 'flex';
                    container.style.alignItems = 'center';
                    container.style.marginBottom = '10px';

                    checkbox.style.marginRight = '10px';

                    container.appendChild(checkbox);
                    container.appendChild(thumbnail);
                    container.appendChild(label);
                    poseSelectContainer.appendChild(container);

                    checkbox.addEventListener('change', (ev) => {
                        if (ev.target.checked) {
                            displayReferencePose(png);
                        } else {
                            removeReferencePose(png);
                        }
                        updateClearAllButtonState();
                    });
                });
            })
            .catch(error => {
                console.error('Error fetching Reference Poses:', error);
                poseSelectContainer.innerHTML = '<p>Error loading Reference Poses</p>';
            });
    }

    function fetchSbReferences() {
        fetch('/api/sb_references', { cache: 'no-store' })
            .then(response => response.json())
            .then(jpgFiles => {
                sbReferenceFiles = jpgFiles;
            })
            .catch(error => {
                console.error('Error fetching sb_references:', error);
            });
    }

    function displayReferencePose(png) {
        if (document.getElementById(`pose-wrapper-${png}`)) {
            return; // already displayed
        }
        const wrapper = document.createElement('div');
        wrapper.className = 'reference-image-wrapper';
        wrapper.id        = `pose-wrapper-${png}`;

        const baseName        = png.replace(/^final_/, '').replace(/\.png$/i, '');
        const sbReferenceName = `${baseName}.jpg`;
        const sbRefExists     = sbReferenceFiles.includes(sbReferenceName);

        // Pose caption + image
        const poseCaption = document.createElement('div');
        poseCaption.className= 'caption';
        poseCaption.textContent= png;

        const poseImg   = document.createElement('img');
        poseImg.src     = `/reference_poses/${png}`;
        poseImg.alt     = png;
        poseImg.onload  = () => { /* loaded */ };
        poseImg.onerror = () => {
            console.error(`Failed to load Reference Pose: ${png}`);
            poseImg.alt = 'Failed to load image.';
        };

        wrapper.appendChild(poseCaption);
        wrapper.appendChild(poseImg);

        // If sb_reference exists
        if (sbRefExists) {
            const sbCaption  = document.createElement('div');
            sbCaption.className= 'caption';
            sbCaption.textContent= sbReferenceName;

            const sbImg   = document.createElement('img');
            sbImg.src     = `/sb_references/${sbReferenceName}`;
            sbImg.alt     = sbReferenceName;
            sbImg.onload  = () => { /* loaded */ };
            sbImg.onerror = () => {
                console.error(`Failed to load sb_reference Image: ${sbReferenceName}`);
                sbImg.alt = 'Failed to load image.';
            };

            wrapper.appendChild(sbCaption);
            wrapper.appendChild(sbImg);
        } else {
            const noSbRef = document.createElement('div');
            noSbRef.className= 'caption';
            noSbRef.textContent= 'No corresponding sb_reference found.';
            wrapper.appendChild(noSbRef);
        }
        referenceImageContainer.appendChild(wrapper);
    }

    function removeReferencePose(png) {
        const wrapper = document.getElementById(`pose-wrapper-${png}`);
        if (wrapper) wrapper.remove();
    }

    function clearAllSelectedPoses() {
        const checkedCheckboxes = poseSelectContainer.querySelectorAll('input[type="checkbox"]:checked');
        checkedCheckboxes.forEach(checkbox => {
            checkbox.checked = false;
            removeReferencePose(checkbox.value);
        });
        updateClearAllButtonState();
    }
    function updateClearAllButtonState() {
        const anyChecked = (poseSelectContainer.querySelectorAll('input[type="checkbox"]:checked').length > 0);
        clearAllBtn.disabled = !anyChecked;
    }
    clearAllBtn.addEventListener('click', clearAllSelectedPoses);
    updateClearAllButtonState();


    /**************************************
     * 7) NOTE-TAKING & ALL NOTES (DATA VIEWER)
     **************************************/
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
                showNotification('Failed to load notes for this frame.', true);
            });
    }

    function displayNotes(notes) {
        const existingNotes = notesList.querySelectorAll('.note-item, .no-notes');
        existingNotes.forEach(n => n.remove());

        if (notes.length === 0) {
            const noNotes = document.createElement('p');
            noNotes.className= 'no-notes';
            noNotes.textContent= 'No notes for this frame.';
            notesList.appendChild(noNotes);
            return;
        }
        // Create note elements
        notes.forEach(note => {
            const noteDiv = document.createElement('div');
            noteDiv.className    = 'note-item';
            noteDiv.dataset.noteId = note.id;

            const timestampDiv  = document.createElement('div');
            timestampDiv.className = 'note-timestamp';
            const dateObj       = new Date(note.timestamp);
            timestampDiv.textContent = dateObj.toLocaleString();

            const contentDiv    = document.createElement('div');
            contentDiv.className= 'note-content';
            contentDiv.textContent = note.content;

            const deleteBtn     = document.createElement('button');
            deleteBtn.textContent = 'Delete';
            deleteBtn.addEventListener('click', () => {
                deleteNote(note.id);
            });

            noteDiv.appendChild(timestampDiv);
            noteDiv.appendChild(contentDiv);
            noteDiv.appendChild(deleteBtn);
            notesList.appendChild(noteDiv);
        });
    }

    function deleteNote(noteId) {
        if (!currentGifName || !currentFrameName) {
            showNotification('No frame selected.', true);
            return;
        }
        const confirmDelete = confirm('Are you sure you want to delete this note?');
        if (!confirmDelete) return;

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
                showNotification('Note deleted successfully.', false);
                fetchAndDisplayNotes(currentGifName, currentFrameName);
                // If AllNotes tab is active, refresh it too
                const activeTab = document.querySelector('.tabcontent.active');
                if (activeTab && activeTab.id === 'AllNotes') {
                    fetchAndDisplayAllNotes();
                }
            })
            .catch(error => {
                console.error('Error deleting note:', error);
                if (error.error) {
                    showNotification(`Error: ${error.error}`, true);
                } else {
                    showNotification('An unexpected error occurred.', true);
                }
            });
    }

    function clearNotesSection() {
        const existingNotes = notesList.querySelectorAll('.note-item, .no-notes');
        existingNotes.forEach(n => n.remove());
        const placeholder = document.createElement('p');
        placeholder.className = 'no-notes';
        placeholder.textContent = 'No GIF selected.';
        notesList.appendChild(placeholder);
    }

    // Save Note Button
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
        const data = { content: noteContent };
        fetch(`/api/gifs/${encodeURIComponent(currentGifName)}/frames/${encodeURIComponent(currentFrameName)}/notes`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
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
                noteInput.value = '';
                showNotification('Note saved successfully.', false);
                fetchAndDisplayNotes(currentGifName, currentFrameName);
                // Refresh AllNotes if active
                const activeTab = document.querySelector('.tabcontent.active');
                if (activeTab && activeTab.id === 'AllNotes') {
                    fetchAndDisplayAllNotes();
                }
            })
            .catch(error => {
                console.error('Error saving note:', error);
                if (error.error) {
                    showNotification(`Error: ${error.error}`, true);
                } else {
                    showNotification('An unexpected error occurred while saving the note.', true);
                }
            });
    });

    /**************************************
     * 8) ALL NOTES (DATA VIEWER)
     **************************************/
    function fetchAndDisplayAllNotes() {
        console.log('fetchAndDisplayAllNotes called.');
        updateAllNotesBtn.disabled  = true;
        const originalBtnText       = updateAllNotesBtn.textContent;
        updateAllNotesBtn.textContent= 'Updating...';
        allNotesList.innerHTML      = '<p>Loading all notes...</p>';

        fetch('/api/notes', { cache: 'no-store' })
            .then(response => {
                console.log('fetchAndDisplayAllNotes: Received response status', response.status);
                if (!response.ok) throw new Error('Failed to fetch all notes.');
                return response.json();
            })
            .then(allNotes => {
                console.log('All Notes fetched:', allNotes);
                displayAllNotes(allNotes);
                updateAllNotesBtn.textContent = originalBtnText;
                updateAllNotesBtn.disabled     = false;
                showNotification('Notes updated successfully.', false);
            })
            .catch(error => {
                console.error('Error fetching all notes:', error);
                allNotesList.innerHTML = '<p>Error loading all notes.</p>';
                showNotification('Failed to load all notes.', true);
                updateAllNotesBtn.textContent = originalBtnText;
                updateAllNotesBtn.disabled     = false;
            });
    }

    function displayAllNotes(allNotes) {
        console.log('displayAllNotes called with:', allNotes);
        allNotesList.innerHTML = '';
        if (!allNotes || Object.keys(allNotes).length === 0) {
            allNotesList.innerHTML = '<p>No notes available.</p>';
            showNotification('No notes available.', false);
            return;
        }
        for (const [gifName, frames] of Object.entries(allNotes)) {
            const gifContainer = document.createElement('div');
            gifContainer.className = 'gif-container';
            gifContainer.style.marginBottom = '20px';

            const gifHeader = document.createElement('h3');
            gifHeader.textContent = `GIF: ${gifName}`;
            gifContainer.appendChild(gifHeader);

            for (const [frameName, notes] of Object.entries(frames)) {
                if (notes.length === 0) continue;
                const frameContainer = document.createElement('div');
                frameContainer.className = 'frame-container';
                frameContainer.style.marginLeft   = '20px';
                frameContainer.style.marginBottom = '10px';

                const frameHeader = document.createElement('h4');
                frameHeader.textContent = `Frame: ${frameName}`;
                frameContainer.appendChild(frameHeader);

                const notesListElement = document.createElement('ul');
                notesListElement.style.listStyleType = 'disc';
                notesListElement.style.marginLeft    = '20px';

                notes.forEach(note => {
                    const noteItem = document.createElement('li');
                    noteItem.textContent = `${note.content} (Added on ${new Date(note.timestamp).toLocaleString()})`;
                    notesListElement.appendChild(noteItem);
                });

                frameContainer.appendChild(notesListElement);
                gifContainer.appendChild(frameContainer);
            }
            allNotesList.appendChild(gifContainer);
        }
        if (allNotesList.innerHTML.trim() === '') {
            allNotesList.innerHTML = '<p>No notes available.</p>';
        }
    }

    /**************************************
     * 9) NOTIFICATIONS
     **************************************/
    function showNotification(message, isError = false) {
        notification.textContent         = message;
        notification.style.backgroundColor = isError ? '#f44336' : '#4CAF50';
        notification.style.display       = 'block';
        notification.classList.add('show');

        setTimeout(() => {
            notification.classList.remove('show');
            notification.classList.add('hide');
            setTimeout(() => {
                notification.style.display = 'none';
                notification.classList.remove('hide');
            }, 500);
        }, 3000);
    }

    /**************************************
     * 10) KEYBOARD NAVIGATION (DATA VIEWER)
     **************************************/
    document.addEventListener('keydown', (event) => {
        // Don't interfere if the user is typing in an input or textarea
        const tag = event.target.tagName.toLowerCase();
        if (tag === 'input' || tag === 'textarea') return;
      
        // Which tab is active?
        const dataViewerActive = document.getElementById('DataViewer').classList.contains('active');
        const mocapActive      = document.getElementById('MoCap').classList.contains('active');
      
        // Handle arrow keys and spacebar for the active tab
        if (dataViewerActive) {
          switch (event.key) {
            case 'ArrowLeft':
              event.preventDefault();
              // Move backward one frame if possible
              if (currentFrameIndex > 0) {
                currentFrameIndex--;
                updateFrame();
                updateControls();       // re-check button states
                if (isPlaying) {
                  resetPlayInterval(); // optional: keep playback in sync
                }
              }
              break;
      
            case 'ArrowRight':
              event.preventDefault();
              // Move forward one frame if possible
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
              // Toggle play/pause with spacebar
              event.preventDefault();
              togglePlayPause(); // your Data Viewer’s play/pause function
              break;
      
            default:
              // Do nothing for other keys
              break;
          }
        } 
        else if (mocapActive) {
          switch (event.key) {
            case 'ArrowLeft':
              event.preventDefault();
              if (mocapCurrentFrameIndex > 0) {
                mocapCurrentFrameIndex--;
                mocapUpdateFrame();
                mocapUpdateControls(); // be sure you have this function!
                if (mocapIsPlaying) {
                  // If you have a "mocapResetPlayInterval()", call it here
                }
              }
              break;
      
            case 'ArrowRight':
              event.preventDefault();
              if (mocapCurrentFrameIndex < mocapPreloadedImages.length - 1) {
                mocapCurrentFrameIndex++;
                mocapUpdateFrame();
                mocapUpdateControls();
                if (mocapIsPlaying) {
                  // e.g., mocapResetPlayInterval()
                }
              }
              break;
      
            case ' ':
              event.preventDefault();
              // Toggle MoCap play/pause
              // If you have a dedicated function, e.g.:
              //   mocapTogglePlayPause();
              // Otherwise, replicate start/stop logic:
              if (mocapIsPlaying) {
                mocapStopPlayback();
              } else if (mocapPreloadedImages.length > 0) {
                mocapStartPlayback();
              }
              mocapUpdateControls();
              break;
      
            default:
              break;
          }
        }
      });
      

    /**************************************
     * 11) INITIAL FETCH CALLS
     **************************************/
    fetchGifs();          // Data Viewer
    fetchReferencePoses(); 
    fetchSbReferences();  

    // The MoCap tab’s fetch is triggered 
    // by calling fetchMoCapGifs() inline (e.g., 
    // in openTab, or when tab is activated).
    
    // If you prefer to load them immediately:
    // fetchMoCapGifs();
});
