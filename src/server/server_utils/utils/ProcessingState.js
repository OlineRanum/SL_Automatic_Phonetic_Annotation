// utils/ProcessingState.js

class ProcessingState {
    constructor() {
        if (!ProcessingState.instance) {
            this._isProcessing = false;
            ProcessingState.instance = this;
            console.log('ProcessingState initialized.');
        }
        return ProcessingState.instance;
    }

    get isProcessing() {
        return this._isProcessing;
    }

    set isProcessing(value) {
        console.log(`Processing state changed to: ${value}`);
        this._isProcessing = value;
    }
}

// Remove Object.freeze to allow state changes
const instance = new ProcessingState();
// Object.freeze(instance); // Removed this line

module.exports = instance;

