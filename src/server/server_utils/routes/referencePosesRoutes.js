// routes/referencePosesRoutes.js
const express = require('express');
const router = express.Router();
const referencePosesController = require('../controllers/referencePosesController');

router.get('/', referencePosesController.listReferencePoses);

module.exports = router;
