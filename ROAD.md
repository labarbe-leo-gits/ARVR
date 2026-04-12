# AR/VR Finger Shape Manipulation Roadmap

## 1. Setup and environment

- Install Python 3.11/3.12/3.14 and enable `Add Python to PATH`
- Create a virtual environment in the project folder:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- Install required packages:
  ```powershell
  python -m pip install opencv-python mediapipe
  ```

## 2. Prototype webcam capture

- Build a simple script to open the webcam using OpenCV
- Display live video frames in a window
- Confirm the camera feed works before adding tracking

## 3. Add hand and finger detection

- Integrate MediaPipe Hands for real-time hand landmark detection
- Track key points like index fingertip, thumb tip, and wrist
- Draw landmark connections on the camera frame

## 4. Define gesture controls

- Detect basic gestures such as:
  - finger point / drag
  - pinch / release
  - open palm / fist
- Map gestures to shape actions:
  - move
  - scale
  - rotate
  - select

## 5. Render interactive shapes

- Draw 2D shapes on the camera overlay:
  - circle
  - rectangle
  - line
- Create a shape object model to store position, size, and rotation
- Update shapes in real time based on finger gestures

## 6. Add interaction modes

- Start with a single shape and one interaction mode
- Expand to multi-shape support later:
  - select a shape by pointing or pinching near it
  - drag to move
  - pinch distance to scale
  - two-finger twist to rotate

## 7. Polish and UX

- Add visual feedback for selected shape and active gesture
- Stabilize tracking with smoothing or filtering
- Add instructions or mode hints on-screen
- Improve lighting and camera guidance for more reliable detection

## 8. Next enhancements

- Add colored shapes and object labels
- Add multi-hand support
- Build a simple GUI with `pygame`, `PyQt`, or a browser frontend
- Upgrade to 3D shape manipulation in Unity or WebGL

## 9. Project goal

- Deliver a working Python prototype that uses webcam-based finger tracking to move and manipulate shapes in real time.
- Keep the first version simple, then iterate with better gestures and user controls.
