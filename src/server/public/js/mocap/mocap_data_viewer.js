// main_mocap.js
import { showNotification } from '../notifications/main_notifications.js';
import { preloadBatch, startPlayback, stopPlayback, resetPlayInterval } from '../viewer_utils.js';

export function initMoCapDataViewer() {
    const mocapGifSelect       = document.getElementById('mocap-gif-select');
    const mocapGifImage        = document.getElementById('mocap-gif-image');
    const mocapFrameInput      = document.getElementById('mocap-frame-input');
    const mocapSaveFramesBtn   = document.getElementById('mocap-save-frames-btn');
    const mocapRemoveInput     = document.getElementById('mocap-remove-input');
    const mocapRemoveFramesBtn = document.getElementById('mocap-remove-frames-btn');
    const indexListContainer   = document.getElementById('mocap-saved-frames-list');

    const mocapLoadingIndicator= document.getElementById('mocap-loading');
    const mocapProgressBar     = document.getElementById('mocap-progress-bar');
    const mocapFrameInfo       = document.getElementById('mocap-frame-info');
    const mocapPrevBtn         = document.getElementById('mocap-prev-btn');
    const mocapNextBtn         = document.getElementById('mocap-next-btn');
    const mocapTogglePlayBtn   = document.getElementById('mocap-toggle-play-btn');
    const mocapFrameRateInput  = document.getElementById('mocap-frame-rate');

    let mocapCurrentGifName    = '';
    let mocapCurrentFrames     = [];
    let mocapPreloadedImages   = [];
    let mocapCurrentFrameIndex = 0;

    const mocapIsPlayingRef    = { value: false };
    const mocapPlayIntervalRef = { value: null };
    let mocapFrameRate         = 200;
    const mocapPreloadBatchSize= 2000;

    // We'll store selected frame indices (fetched from server) in a local object
    let frameIndices = {};


    // ---------- (B) HANDLE MOCAP GIF SELECTION ----------
    mocapGifSelect.addEventListener('change', () => {
        const selectedGif = mocapGifSelect.value;
        if (!selectedGif) {
            resetMocapViewer();
            return;
        }
        mocapCurrentGifName = selectedGif;
        stopMocapPlaybackImmediate();

        // Show loading
        mocapLoadingIndicator.style.display = 'block';
        mocapProgressBar.style.width = '0%';

        // Fetch frames
        fetch(`/api/graphics/mocap_gifs/${encodeURIComponent(selectedGif)}/frames`, { cache: 'no-store' })
            .then(r => {
                if (!r.ok) throw new Error('Frames not found.');
                return r.json();
            })
            .then(frames => {
                if (!frames.length) throw new Error('No frames available for this GIF.');
                mocapCurrentFrames = frames;
                mocapCurrentFrameIndex = 0;
                mocapPreloadedImages = new Array(mocapCurrentFrames.length).fill(null);

                // Start preloading
                preloadBatch(
                  mocapCurrentFrames,
                  mocapPreloadedImages,
                  0,
                  mocapPreloadBatchSize,
                  (percent) => {
                    mocapProgressBar.style.width = `${percent}%`;
                  }
                ).then(endIndex => {
                    mocapUpdateFrame();
                    mocapUpdateControls();
                    if (endIndex >= mocapCurrentFrames.length) {
                        // fully loaded
                        mocapLoadingIndicator.style.display = 'none';
                    } else {
                        // keep preloading in background
                        continueMocapPreloading(endIndex);
                    }
                }).catch(err => {
                    console.error('Error preloading MoCap frames:', err);
                    mocapLoadingIndicator.innerHTML = 'Error loading frames.';
                });
            })
            .catch(err => {
                console.error('Error fetching MoCap frames:', err);
                mocapLoadingIndicator.innerHTML = 'Error loading frames.';
            });

        // Also fetch previously stored frame indexes
        fetchMoCapSelectedFrames();
    });

    function continueMocapPreloading(startIndex) {
        if (startIndex >= mocapCurrentFrames.length) {
            mocapLoadingIndicator.style.display = 'none';
            return;
        }
        preloadBatch(
            mocapCurrentFrames,
            mocapPreloadedImages,
            startIndex,
            mocapPreloadBatchSize,
            (percent) => {
                mocapProgressBar.style.width = `${percent}%`;
            }
        ).then(nextIndex => {
            if (nextIndex >= mocapCurrentFrames.length) {
                mocapLoadingIndicator.style.display = 'none';
            } else {
                continueMocapPreloading(nextIndex);
            }
        }).catch(err => {
            console.error('Error preloading MoCap frames:', err);
            mocapLoadingIndicator.innerHTML = 'Error loading frames.';
        });
    }

    // ---------- (C) UPDATE FRAME ----------
    function mocapUpdateFrame() {
        if (mocapPreloadedImages[mocapCurrentFrameIndex]) {
            mocapGifImage.src = mocapPreloadedImages[mocapCurrentFrameIndex].src;
            mocapFrameInfo.textContent = `Frame: ${mocapCurrentFrameIndex + 1} / ${mocapPreloadedImages.length}`;
        } else {
            mocapGifImage.src = '';
            mocapFrameInfo.textContent = `Frame: ${mocapCurrentFrameIndex + 1} / ${mocapPreloadedImages.length}`;
        }
    }

    // ---------- (D) PLAYBACK ----------
    function mocapStartPlayback() {
        mocapTogglePlayBtn.textContent = 'Stop GIF';
        mocapPlayIntervalRef.value = startPlayback({
            isPlayingRef: mocapIsPlayingRef,
            frameRate: mocapFrameRate,
            currentFrameIndexRef: { value: mocapCurrentFrameIndex },
            preloadedImages: mocapPreloadedImages,
            updateFrame: () => {
                mocapCurrentFrameIndex = mocapPlayIntervalRef.value
                  ? (mocapPlayIntervalRef.value.currentFrameIndexRef?.value ?? mocapCurrentFrameIndex)
                  : mocapCurrentFrameIndex;
                mocapUpdateFrame();
            },
            updateControls: mocapUpdateControls
        });
    }
    function mocapStopPlayback() {
        mocapTogglePlayBtn.textContent = 'Play GIF';
        stopPlayback({ isPlayingRef: mocapIsPlayingRef, playIntervalRef: mocapPlayIntervalRef });
    }
    function stopMocapPlaybackImmediate() {
        mocapIsPlayingRef.value = false;
        if (mocapPlayIntervalRef.value) {
            clearInterval(mocapPlayIntervalRef.value);
            mocapPlayIntervalRef.value = null;
        }
        mocapTogglePlayBtn.textContent = 'Play GIF';
    }

    // ---------- (E) CONTROLS ----------
    function mocapUpdateControls() {
        if (mocapPreloadedImages.length > 0) {
            const isSingleFrame = (mocapPreloadedImages.length === 1);
            mocapPrevBtn.disabled       = (mocapIsPlayingRef.value || mocapCurrentFrameIndex === 0 || isSingleFrame);
            mocapNextBtn.disabled       = (mocapIsPlayingRef.value || mocapCurrentFrameIndex === (mocapPreloadedImages.length - 1) || isSingleFrame);
            mocapTogglePlayBtn.disabled = isSingleFrame;
            mocapTogglePlayBtn.textContent = mocapIsPlayingRef.value ? 'Stop GIF' : 'Play GIF';
        } else {
            // no frames
            mocapPrevBtn.disabled       = true;
            mocapNextBtn.disabled       = true;
            mocapTogglePlayBtn.disabled = true;
            mocapTogglePlayBtn.textContent = 'Play GIF';
        }
    }

    // ---------- (F) BUTTON LISTENERS ----------
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
        if (mocapIsPlayingRef.value) {
            mocapStopPlayback();
        } else if (mocapPreloadedImages.length > 0) {
            mocapStartPlayback();
        }
        mocapUpdateControls();
    });

    mocapFrameRateInput.addEventListener('change', () => {
        const newRate = parseInt(mocapFrameRateInput.value, 10);
        if (!isNaN(newRate) && newRate >= 10 && newRate <= 1000) {
            mocapFrameRate = newRate;
            if (mocapIsPlayingRef.value) {
                stopMocapPlaybackImmediate();
                mocapStartPlayback();
            }
        } else {
            mocapFrameRateInput.value = mocapFrameRate;
            showNotification('Invalid MoCap frame rate. Reset to previous value.', true);
        }
    });

    // ---------- (G) FRAME INDEX STORING ----------
    mocapSaveFramesBtn.addEventListener('click', () => {
        if (!mocapCurrentGifName) {
            alert('No MoCap GIF is selected!');
            return;
        }
        const inputValue = mocapFrameInput.value.trim();
        if (!inputValue) {
            alert('Please enter a single index or a range (e.g. 210 or 211-234).');
            return;
        }
        const baseName = encodeURIComponent(mocapCurrentGifName);
        fetch(`/api/graphics/mocap_gifs/${baseName}/selected_frames`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rangeOrIndex: inputValue })
        })
        .then(r => {
            if (!r.ok) throw new Error('Failed to store frames.');
            return r.json();
        })
        .then(data => {
            console.log('Frames stored:', data.frames);
            mocapFrameInput.value = '';
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
        fetch(`/api/graphics/mocap_gifs/${encoded}/selected_frames?start=${start}&end=${end}`, {
            method: 'DELETE'
        })
        .then(r => {
            if (!r.ok) throw new Error('Failed to remove frames.');
            return r.json();
        })
        .then(data => {
            console.log('Removed frames:', data);
            mocapRemoveInput.value = '';
            fetchMoCapSelectedFrames();
        })
        .catch(err => {
            console.error('Error removing frames:', err);
            alert('Error removing frames. Check console.');
        });
    });

    function fetchMoCapSelectedFrames() {
        if (!mocapCurrentGifName) {
            indexListContainer.innerHTML = '<p>No MoCap GIF selected.</p>';
            frameIndices = {};
            return;
        }
        const baseName = encodeURIComponent(mocapCurrentGifName);
        fetch(`/api/graphics/mocap_gifs/${baseName}/selected_frames`)
            .then(r => r.json())
            .then(indexes => {
                frameIndices = {};
                indexes.forEach(i => (frameIndices[i] = true));
                const ranges = formatIndicesAsRanges(indexes);
                displayMoCapSelectedFrames(ranges);
            })
            .catch(err => {
                console.error('Error fetching MoCap selected frames:', err);
                indexListContainer.innerHTML = '<p>Error loading selected frames.</p>';
                frameIndices = {};
            });
    }

    function displayMoCapSelectedFrames(ranges) {
        if (!ranges || !ranges.length) {
            indexListContainer.innerHTML = '<p>No frames stored.</p>';
            return;
        }
        indexListContainer.innerHTML = '';
        ranges.forEach(range => {
            const rangeItem = document.createElement('div');
            rangeItem.textContent = range;
            rangeItem.style.padding = '5px 10px';
            rangeItem.style.margin = '5px 0';
            rangeItem.style.backgroundColor = '#f0f0f0';
            rangeItem.style.border = '1px solid #ccc';
            rangeItem.style.borderRadius = '5px';

            const rangeContainer = document.createElement('div');
            rangeContainer.style.display = 'flex'; // Use flexbox for layout
            rangeContainer.style.justifyContent = 'space-between'; // Space out items
            rangeContainer.style.alignItems = 'center'; // Align items vertically
            
            const rangeText = document.createElement('span');
            rangeContainer.appendChild(rangeText);
            
            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = 'Delete';
            deleteBtn.style.marginLeft = '10px'; // Add spacing to the left of the button
            deleteBtn.addEventListener('click', () => {
                deleteRange(range);
            });
            
            rangeContainer.appendChild(deleteBtn);
            rangeItem.appendChild(rangeContainer);
            indexListContainer.appendChild(rangeItem);
            
        });
    }

    function deleteRange(range) {
        const [startStr, endStr] = range.split('-');
        const start = parseInt(startStr, 10);
        const end   = endStr ? parseInt(endStr, 10) : start;
        const encoded = encodeURIComponent(mocapCurrentGifName);
        fetch(`/api/graphics/mocap_gifs/${encoded}/selected_frames?start=${start}&end=${end}`, {
            method: 'DELETE'
        })
        .then(r => {
            if (!r.ok) throw new Error('Failed to delete range on the server.');
            return r.json();
        })
        .then(data => {
            console.log(`Range ${range} deleted successfully.`, data);
            // Remove them from local frameIndices
            for (let i = start; i <= end; i++) {
                delete frameIndices[i];
            }
            // Refresh
            const indexes = Object.keys(frameIndices).map(Number);
            const ranges = formatIndicesAsRanges(indexes);
            displayMoCapSelectedFrames(ranges);
        })
        .catch(err => {
            console.error('Error deleting range:', err);
            alert('Failed to delete the range. Check console.');
        });
    }

    function formatIndicesAsRanges(indices) {
        if (!indices || !indices.length) return [];
        indices.sort((a, b) => a - b);
        const ranges = [];
        let start = indices[0];
        let end = start;
        for (let i = 1; i < indices.length; i++) {
            if (indices[i] === end + 1) {
                end = indices[i];
            } else {
                ranges.push(start === end ? `${start}` : `${start}-${end}`);
                start = indices[i];
                end   = start;
            }
        }
        ranges.push(start === end ? `${start}` : `${start}-${end}`);
        return ranges;
    }

    // ---------- (H) KEYBOARD NAV ----------

    document.addEventListener('keydown', (ev) => {
        const tag = ev.target.tagName.toLowerCase();
        if (tag === 'input' || tag === 'textarea') return;

        const mocapViewerActive = document.getElementById('mocapDataViewerTab').classList.contains('active');
        if (!mocapViewerActive) return;

        console.log('Keydown event detected. MoCap active:', mocapViewerActive); // Debugging log


        switch (ev.key) {
            case 'ArrowLeft':
                ev.preventDefault();
                if (mocapCurrentFrameIndex > 0) {
                    mocapCurrentFrameIndex--;
                    mocapUpdateFrame();
                    mocapUpdateControls();
                    if (mocapIsPlayingRef.value) {
                        resetPlayInterval({
                            isPlayingRef: mocapIsPlayingRef,
                            playIntervalRef: mocapPlayIntervalRef,
                            frameRate: mocapFrameRate,
                            currentFrameIndexRef: { value: mocapCurrentFrameIndex },
                            preloadedImages: mocapPreloadedImages,
                            updateFrame: () => {
                                mocapCurrentFrameIndex = mocapPlayIntervalRef.value
                                  ? (mocapPlayIntervalRef.value.currentFrameIndexRef?.value ?? mocapCurrentFrameIndex)
                                  : mocapCurrentFrameIndex;
                                mocapUpdateFrame();
                            },
                            updateControls: mocapUpdateControls
                        });
                    }
                }
                break;
            case 'ArrowRight':
                ev.preventDefault();
                if (mocapCurrentFrameIndex < mocapPreloadedImages.length - 1) {
                    mocapCurrentFrameIndex++;
                    mocapUpdateFrame();
                    mocapUpdateControls();
                    if (mocapIsPlayingRef.value) {
                        resetPlayInterval({
                            isPlayingRef: mocapIsPlayingRef,
                            playIntervalRef: mocapPlayIntervalRef,
                            frameRate: mocapFrameRate,
                            currentFrameIndexRef: { value: mocapCurrentFrameIndex },
                            preloadedImages: mocapPreloadedImages,
                            updateFrame: () => {
                                mocapCurrentFrameIndex = mocapPlayIntervalRef.value
                                  ? (mocapPlayIntervalRef.value.currentFrameIndexRef?.value ?? mocapCurrentFrameIndex)
                                  : mocapCurrentFrameIndex;
                                mocapUpdateFrame();
                            },
                            updateControls: mocapUpdateControls
                        });
                    }
                }
                break;
            case ' ':
                ev.preventDefault();
                // Toggle MoCap play/pause
                if (mocapIsPlayingRef.value) {
                    mocapStopPlayback();
                } else if (mocapPreloadedImages.length > 0) {
                    mocapStartPlayback();
                }
                mocapUpdateControls();
                break;
            default:
                break;
        }
    });

    // ---------- (I) RESET VIEWER ----------
    function resetMocapViewer() {
        stopMocapPlaybackImmediate();
        mocapCurrentGifName      = '';
        mocapCurrentFrames       = [];
        mocapPreloadedImages     = [];
        mocapCurrentFrameIndex   = 0;
        mocapGifImage.src        = '';
        mocapFrameInfo.textContent = 'Frame: 0';
        mocapLoadingIndicator.style.display = 'none';
        mocapProgressBar.style.width = '0%';
        indexListContainer.innerHTML = '<p>No MoCap GIF selected.</p>';
        mocapUpdateControls();
    }



    // ---------- (J) INIT -----------
    fetchMoCapGifs();
}



export function fetchMoCapGifs() {
    const mocapGifSelect       = document.getElementById('mocap-gif-select');

    fetch('/api/graphics/mocap_gifs', { cache: 'no-store' })
        .then(r => r.json())
        .then(gifs => {
            if (!gifs.length) {
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
            mocapGifSelect.innerHTML = '<option value="">Error loading GIFs</option>';
        });
}

