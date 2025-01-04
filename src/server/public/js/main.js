import { initDataViewer } from './dataViewer.js';
import { initMoCap } from './mocap.js';
import { initReferencePoses } from './referencePoses.js';
import { initNotes } from './notes.js';
import { initNotifications } from './notifications.js';

document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing modules...');
    initNotes();
    initNotifications();
    initReferencePoses();
    initDataViewer();
    initMoCap();
    console.log('Modules initialized successfully.');
});
