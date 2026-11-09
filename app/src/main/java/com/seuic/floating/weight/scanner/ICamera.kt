package com.seuic.floating.weight.scanner

interface ICamera {
	/**
	 * This method is invoked shall perform concrete operation to initialize the camera.
	 * CONTRACT: as a result of this method variables mFrameWidth and mFrameHeight MUST be
	 * initialized with the size of the Camera frames that will be delivered to external processor.
	 *
	 * @param width  - the width of this SurfaceView
	 * @param height - the height of this SurfaceView
	 */
	open fun connectCamera(isFlash: Boolean = true): Boolean

	/**
	 * Disconnects and release the particular camera object being connected to this surface view.
	 * Called when syncObject lock is held
	 */
	open fun disconnectCamera()
}
