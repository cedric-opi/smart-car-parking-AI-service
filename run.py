#!/usr/bin/env python3
"""
Car Parking Detection System - Main Entry Point
Run this script to start the parking detection system
"""

import argparse
import logging
import os
import sys

import config
from enhanced_parking_detector import EnhancedParkingDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("parking_detection.log")],
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     🚗  Car Parking Space Detection System  🚗           ║
    ║                                                           ║
    ║     Powered by YOLOv8 & OpenCV                          ║
    ║     Version 2.0                                          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def validate_file(file_path, file_type="file"):
    """Validate file exists and is accessible"""
    if not file_path:
        return False

    if not os.path.exists(file_path):
        logger.error(f"{file_type.capitalize()} not found: {file_path}")
        print(f"❌ Error: {file_type.capitalize()} not found at '{file_path}'")
        print(f"   Please check the path and try again.")
        return False

    if not os.access(file_path, os.R_OK):
        logger.error(f"{file_type.capitalize()} not readable: {file_path}")
        print(f"❌ Error: Cannot read {file_type} at '{file_path}'")
        print(f"   Please check file permissions.")
        return False

    return True


def main():
    """Main entry point for the application"""

    # Print banner
    print_banner()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Advanced Car Parking Space Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
   # Process an image
   python run.py --image carParkImg.jpg

   # Process a video file
   python run.py --video carPark.mp4

   # Process both (image for setup, video for detection)
   python run.py --image carParkImg.jpg --video carPark.mp4

   # Live camera stream (RTSP)
   python run.py --camera rtsp://user:pass@192.168.1.100:554/stream1

   # Live camera stream (HTTP/MJPEG)
   python run.py --camera http://192.168.1.100:8080/video

 Keyboard Shortcuts (during execution):
   D - Detect vehicles and generate reports
   S - Save parking space layout
   R - Reset all parking spaces
   Z - Undo last selection
   Q - Quit application

 For more information, visit: https://github.com/8harath/Car-Parking-Detection
         """,
    )

    parser.add_argument(
        "--image",
        "-i",
        type=str,
        help="Path to parking lot image for space selection",
        default=config.DEFAULT_IMAGE_PATH,
    )

    parser.add_argument(
        "--video", "-v", type=str, help="Path to parking lot video for detection", default=None
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["image", "video", "camera", "both"],
        default="image",
        help="Processing mode: image, video, camera, or both",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument("--version", action="version", version="Car Parking Detection System v2.0")

    parser.add_argument(
        "--camera",
        "-c",
        type=str,
        default=None,
        help=(
            "URL of the live camera stream for real-time parking detection. "
            "Supports RTSP, HTTP, and MJPEG streams. "
            "Examples: "
            "rtsp://user:pass@192.168.1.100:554/stream1  "
            "http://192.168.1.100:8080/video"
        ),
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help=(
            "Capture a frame from the camera and open the interactive slot-editor "
            "to define parking spaces. Use this BEFORE running detection. "
            "Example: python run.py --camera http://<ip>/video --setup"
        ),
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate inputs
    logger.info("Starting Car Parking Detection System...")
    print("\n🚀 Starting application...")

    # Determine mode
    if args.camera:
        mode = "camera"
    elif args.video and not args.image:
        mode = "video"
    elif args.image and not args.video:
        mode = "image"
    elif args.image and args.video:
        mode = "both"
    else:
        mode = args.mode

    # Validate files based on mode
    if mode in ["image", "both"]:
        if not validate_file(args.image, "image"):
            sys.exit(1)
        print(f"✓ Image file found: {args.image}")

    if mode in ["video", "both"]:
        if not args.video:
            logger.error("Video mode requires --video argument")
            print("❌ Error: Video mode requires a video file path")
            print("   Use: python run.py --video path/to/video.mp4")
            sys.exit(1)
        if not validate_file(args.video, "video"):
            sys.exit(1)
        print(f"✓ Video file found: {args.video}")

    if mode == "camera":
        camera_url = args.camera or config.CAMERA_URL
        if not camera_url:
            logger.error("Camera mode requires --camera <url> argument or CAMERA_URL in config.py")
            print("❌ Error: Camera mode requires a stream URL")
            print("   Use: python run.py --camera rtsp://user:pass@<ip>:<port>/stream")
            sys.exit(1)
        print(f"✓ Camera URL: {camera_url}")

    try:
        # Initialize detector
        logger.info("Initializing parking detector...")
        print("\n⚙️  Initializing detector...")

        camera_url = args.camera if mode == "camera" else None

        detector = EnhancedParkingDetector(
            image_path=args.image if mode in ["image", "both"] else None,
            video_path=args.video if mode in ["video", "both"] else None,
            camera_url=camera_url,
        )

        print("✓ Detector initialized successfully")

        # ── Camera setup mode (must run before detection) ──────────────────
        if mode == "camera" and args.setup:
            logger.info(f"Running camera setup for: {camera_url}")
            print(f"\n🛠️  Camera setup mode: {camera_url}")
            detector.setup_from_camera()
            return 0

        # Process based on mode
        if mode == "image" or mode == "both":
            logger.info(f"Processing image: {args.image}")
            print(f"\n📸 Processing image: {args.image}")
            print("\n" + "=" * 60)
            print("KEYBOARD SHORTCUTS:")
            print("=" * 60)
            print("  D - Detect vehicles & generate reports")
            print("  S - Save parking layout")
            print("  R - Reset all selections")
            print("  Z - Undo last selection")
            print("  Q - Quit application")
            print("=" * 60 + "\n")

            detector.process_image()

        if mode == "video":
            logger.info(f"Processing video: {args.video}")
            print(f"\n🎥 Processing video: {args.video}")
            print("\nPress 'Q' to quit\n")

            detector.process_video()

        if mode == "camera":
            logger.info(f"Processing live camera stream: {camera_url}")
            print(f"\n📹 Starting live camera detection: {camera_url}")

            detector.process_camera()

        # Success message
        logger.info("Processing completed successfully")
        print("\n✅ Processing completed successfully!")
        print(f"📁 Reports saved in: {config.REPORTS_DIR}/")
        print(f"💾 CSV data saved in: {config.DATA_DIR}/{config.CSV_FILE}")

        return 0

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n\n⚠️  Application interrupted by user")
        print("Goodbye! 👋")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check the file path and try again.")
        return 1

    except IOError as e:
        logger.error(f"IO Error: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check file permissions and disk space.")
        return 1

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error occurred: {e}")
        print("Check parking_detection.log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
