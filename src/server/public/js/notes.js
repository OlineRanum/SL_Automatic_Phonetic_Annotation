import { showNotification } from './notifications.js';

export function initNotes() {
    // -------- Note-Taking Elements (Data Viewer Only) --------
    const noteInput = document.getElementById('note-input');
    const saveNoteBtn = document.getElementById('save-note-btn');
    const notesList = document.getElementById('notes-list');
    let currentGifName = ''; // Initialize with an empty string or null
    let currentFrameName = 'frame_0000.png'; // Initialize with an empty string or null


    // -------- All Notes Elements (Data Viewer Only) --------
    const allNotesList = document.getElementById('all-notes-list');
    const updateAllNotesBtn = document.getElementById('update-all-notes-btn');
    fetchAndDisplayAllNotes(); // Fetch and display all notes when the page loads

    document.addEventListener('frameUpdated', (event) => {
        const { gifName, frameName } = event.detail;
        currentGifName = gifName;
        currentFrameName = frameName;
    });
    
    document.addEventListener('gifUpdated', (event) => {
        const { gifName } = event.detail;
        currentGifName = gifName;
        currentFrameName = null; // Reset frame when a new GIF is selected
    });


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
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw err; });
                }
                return response.json();
            })
            .then(newNote => {
                noteInput.value = '';
                showNotification('Note saved successfully.', false);
                fetchAndDisplayNotes(currentGifName, currentFrameName);
            })
            .catch(error => {
                console.error('Error saving note:', error);
                showNotification('An unexpected error occurred while saving the note.', true);
            });
    });
    
    function clearNotesSection() {
        const existingNotes = notesList.querySelectorAll('.note-item, .no-notes');
        existingNotes.forEach(n => n.remove());
        const placeholder = document.createElement('p');
        placeholder.className = 'no-notes';
        placeholder.textContent = 'No GIF selected.';
        notesList.appendChild(placeholder);
    }
    updateAllNotesBtn.addEventListener('click', fetchAndDisplayAllNotes); // Add this event listener

    function fetchAndDisplayAllNotes() {
        updateAllNotesBtn.disabled  = true;
        const originalBtnText       = updateAllNotesBtn.textContent;
        updateAllNotesBtn.textContent= 'Updating...';
        allNotesList.innerHTML      = '<p>Loading all notes...</p>';

        fetch('/api/notes', { cache: 'no-store' })
            .then(response => {
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

}

export function deleteNote(noteId, gifName, frameName) {
    
    console.log(`Deleting note ${noteId} from GIF ${gifName} and frame ${frameName}`);
    if (!gifName || !frameName) {
        showNotification('No frame selected.', true);
        return;
    }

    const confirmDelete = confirm('Are you sure you want to delete this note?');
    if (!confirmDelete) return;

    fetch(`/api/gifs/${encodeURIComponent(gifName)}/frames/${encodeURIComponent(frameName)}/notes/${encodeURIComponent(noteId)}`, {
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
            fetchAndDisplayNotes(gifName, frameName);
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



export function displayNotes(notes, gifName, frameName) {
    const notesList = document.getElementById('notes-list'); // Ensure this element exists
    const existingNotes = notesList.querySelectorAll('.note-item, .no-notes');
    existingNotes.forEach(n => n.remove());

    if (notes.length === 0) {
        const noNotes = document.createElement('p');
        noNotes.className = 'no-notes';
        noNotes.textContent = 'No notes for this frame.';
        notesList.appendChild(noNotes);
        return;
    }

    notes.forEach(note => {
        const noteDiv = document.createElement('div');
        noteDiv.className = 'note-item';
        noteDiv.dataset.noteId = note.id;

        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'note-timestamp';
        timestampDiv.textContent = new Date(note.timestamp).toLocaleString();

        const contentDiv = document.createElement('div');
        contentDiv.className = 'note-content';
        contentDiv.textContent = note.content;

        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = 'Delete';
        deleteBtn.addEventListener('click', () => {
            deleteNote(note.id, gifName, frameName);
        });

        noteDiv.appendChild(timestampDiv);
        noteDiv.appendChild(contentDiv);
        noteDiv.appendChild(deleteBtn);
        notesList.appendChild(noteDiv);
    });
}


export function fetchAndDisplayNotes(gifName, frameName) {
    if (!gifName || !frameName) {
        clearNotesSection();
        return;
    }
    fetch(`/api/gifs/${encodeURIComponent(gifName)}/frames/${encodeURIComponent(frameName)}/notes`, { cache: 'no-store' })
        .then(response => response.json())
        .then(notes => {
            displayNotes(notes, gifName, frameName);
        })
        .catch(error => {
            console.error('Error fetching notes:', error);
            showNotification('Failed to load notes for this frame.', true);
        });
}
