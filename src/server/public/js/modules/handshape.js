export function initHandshapeViewer() {
    // -------- Reference Poses Viewer Elements --------
    const poseSelectContainer    = document.getElementById('pose-select-container');
    const referenceImageContainer= document.getElementById('reference-poses-container');
    const poseLoading            = document.getElementById('pose-loading');
    const clearAllBtn            = document.getElementById('clear-all-btn');
    // Reference Poses / sb_references
    let referencePoses     = [];
    let sbReferenceFiles   = [];
    fetchReferencePoses(); 
    fetchSbReferences();  
    
    function fetchReferencePoses() {
        fetch('/api/graphics/reference_poses', { cache: 'no-store' })
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
                    thumbnail.src   = `/graphics/reference_poses/${png}`;
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
        fetch('/api/graphics/sb_references', { cache: 'no-store' })
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
        poseImg.src     = `/graphics/reference_poses/${png}`;
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
            sbImg.src     = `/graphics/sb_references/${sbReferenceName}`;
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


}
