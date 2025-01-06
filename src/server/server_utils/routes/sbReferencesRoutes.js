// routes/sbReferencesRoutes.js
const express = require('express');
const router = express.Router();
const sbReferencesController = require('../controllers/sbReferencesController');

router.get('/', sbReferencesController.listSbReferences);

module.exports = router;
