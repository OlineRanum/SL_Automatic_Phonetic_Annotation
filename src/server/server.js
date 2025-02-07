// server.js
const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const app = express();




const processMocapRoutes = require('./server_utils/routes/processMocapRoutes');

// Middleware to parse JSON bodies
app.use(bodyParser.json());


app.use('/api/process_mocap', processMocapRoutes);

// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));

// Import routes
const gifsRoutes = require('./server_utils/routes/gifsRoutes');
const referencePosesRoutes = require('./server_utils/routes/referencePosesRoutes');
const sbReferencesRoutes = require('./server_utils/routes/sbReferencesRoutes');
const notesRoutes = require('./server_utils/routes/notesRoutes');
const mocapGifsRoutes = require('./server_utils/routes/mocapGifsRoutes');
const dataMocapRoutes = require('./server_utils/routes/dataRoutes');
// Mount routes
app.use('/api/graphics/gifs', gifsRoutes);
app.use('/api/graphics/reference_poses', referencePosesRoutes);
app.use('/api/graphics/sb_references', sbReferencesRoutes);
app.use('/api/notes', notesRoutes);
app.use('/api/graphics/mocap_gifs', mocapGifsRoutes);
app.use('/api/data', dataMocapRoutes);



// Start the server
const PORT = process.env.PORT || 2006;
app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});
