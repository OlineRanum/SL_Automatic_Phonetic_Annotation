import { fetchAndDisplayNotes } from './notes.js';
import { deleteNote } from './notes.js';
import { showNotification } from './notifications.js';

export function initDataViewer() {
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

    let currentGifName     = 'None';
    let currentFrameName   = 'frame_0000.png';
    let currentFrames      = [];
    let preloadedImages    = [];
    let currentFrameIndex  = 0;
    let isPlaying          = false;
    let playInterval       = null;
    let frameRate          = 200;    // 200ms per frame (5 fps)
    
    const preloadBatchSize = 1000;    // Number of frames to preload at a time
    
    fetchGifs();          // Data Viewer

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

            const gifUpdateEvent = new CustomEvent('gifUpdated', {
                detail: {
                    gifName: selectedGif,
                    frameName: currentFrameName,
                },
            });

            document.dispatchEvent(gifUpdateEvent);

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
            const gifUpdateEvent = new CustomEvent('gifUpdated', {
                detail: {
                    gifName: currentGifName,
                    frameName: currentFrameName,
                },
            });
            document.dispatchEvent(gifUpdateEvent);
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
                
                // Dispatch an event to notify listeners
                const frameUpdateEvent = new CustomEvent('frameUpdated', {
                    detail: {
                        gifName: currentGifName,
                        frameName: currentFrameName,
                    },
                });
                document.dispatchEvent(frameUpdateEvent);

                fetchAndDisplayNotes(currentGifName, currentFrameName);
            } else {
                // If not preloaded
                gifImage.src = '';
                frameInfo.textContent = `Frame: ${currentFrameIndex + 1} / ${preloadedImages.length}`;
                const frameUpdateEvent = new CustomEvent('frameUpdated', {
                    detail: {
                        gifName: currentGifName,
                        frameName: currentFrameName,
                    },
                });
                document.dispatchEvent(frameUpdateEvent);
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
          

}

document.addEventListener('DOMContentLoaded', () => {
    initDataViewer();
});