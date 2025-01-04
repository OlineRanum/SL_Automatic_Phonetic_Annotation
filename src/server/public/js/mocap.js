export function initMoCap() {
    const mocapGifSelect = document.getElementById('mocap-gif-select');
    const mocapGifImage = document.getElementById('mocap-gif-image');
    let mocapCurrentGifName     = '';
    let mocapCurrentFrames      = [];
    let mocapPreloadedImages    = [];
    let mocapCurrentFrameIndex  = 0;
    let mocapIsPlaying          = false;
    let mocapPlayInterval       = null;
    let mocapFrameRate          = 200;  // also 200ms
    const mocapPreloadBatchSize = 2000;
    let frameIndices = {};

    const indexListContainer = document.getElementById('mocap-saved-frames-list');


    const mocapFrameInput      = document.getElementById('mocap-frame-input');
    const mocapSaveFramesBtn   = document.getElementById('mocap-save-frames-btn');
    const mocapSavedFramesList = document.getElementById('mocap-saved-frames-list');
    const mocapRemoveInput    = document.getElementById('mocap-remove-input');
    const mocapRemoveFramesBtn= document.getElementById('mocap-remove-frames-btn');

    const mocapLoadingIndicator = document.getElementById('mocap-loading');
    const mocapProgressBar      = document.getElementById('mocap-progress-bar');
    const mocapFrameInfo        = document.getElementById('mocap-frame-info');
    const mocapPrevBtn          = document.getElementById('mocap-prev-btn');
    const mocapNextBtn          = document.getElementById('mocap-next-btn');
    const mocapTogglePlayBtn    = document.getElementById('mocap-toggle-play-btn');
    const mocapFrameRateInput   = document.getElementById('mocap-frame-rate');

    fetchMoCapGifs();
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
            frameIndices = {}; // Clear frameIndices as well
            return;
        }
    
        const baseName = encodeURIComponent(mocapCurrentGifName);
        fetch(`/api/mocap_gifs/${baseName}/selected_frames`)
            .then(response => response.json())
            .then(indexes => {
                // Populate frameIndices with the fetched indices
                frameIndices = {};
                indexes.forEach(index => frameIndices[index] = true);
    
                const ranges = formatIndicesAsRanges(indexes); // Format indexes into ranges
                displayMoCapSelectedFrames(ranges);
            })
            .catch(error => {
                console.error('Error fetching MoCap selected frames:', error);
                mocapSavedFramesList.innerHTML = '<p>Error loading selected frames.</p>';
                frameIndices = {}; // Clear frameIndices on error
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
        const encoded = encodeURIComponent(mocapCurrentGifName);
    
        // Send DELETE request to the server for the range
        fetch(`/api/mocap_gifs/${encoded}/selected_frames?start=${start}&end=${end || start}`, {
            method: 'DELETE'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to delete range on the server.');
            }
            return response.json();
        })
        .then(data => {
            console.log(`Range ${range} deleted successfully on the server.`, data);
    
            // Update the local list after server deletion
            if (end === undefined) {
                // Single index
                delete frameIndices[start];
            } else {
                // Range of indices
                for (let i = start; i <= end; i++) {
                    delete frameIndices[i];
                }
            }
    
            // Refresh the displayed list of indices
            updateDisplayedIndices();
        })
        .catch(err => {
            console.error('Error deleting range:', err);
            alert('Failed to delete the range. Check the console for details.');
        });
    }
    


    
    
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
              if (mocapCurrentFrameIndex > 0) {
                mocapCurrentFrameIndex--;
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
              if (mocapCurrentFrameIndex < preloadedImages.length - 1) {
                mocapCurrentFrameIndex++;
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
    
}
