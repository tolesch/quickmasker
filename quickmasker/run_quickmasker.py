"""
Entry point script to run the QuickMasker annotation tool.
"""

import sys
import logging
import argparse

try:
    from quickmasker.config import load_and_validate_config
    from quickmasker.main_controller import AnnotationController
except ModuleNotFoundError as e:
    print(f"ERROR: Could not import QuickMasker modules: {e}")
    print("This usually means the QuickMasker package is not installed correctly in your Python environment,")
    print("or you are not running this from an environment where it's installed.")
    print("Please try reinstalling the package from the project root directory using: pip install -e .")
    sys.exit(1)

DEFAULT_CONFIG_PATH = "config.yaml" # Relative to where the command is run

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("QuickMaskerApp")


def main():
    """Loads config, initializes controller, and runs the application."""
    log.info("--- QuickMasker Application Starting ---")

    ### Arg Parsing
    parser = argparse.ArgumentParser(description="QuickMasker SAM Annotation Tool")
    parser.add_argument(
        "-c", "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the configuration YAML file (default: {DEFAULT_CONFIG_PATH})"
    )
    args = parser.parse_args()

    ### Load Config
    log.info(f"Loading configuration from: {args.config}")
    app_config = load_and_validate_config(args.config)
    if app_config is None:
        log.critical("Failed to load or validate configuration. Exiting.")
        sys.exit(1)

    ### Initialize and Run Controller
    try:
        controller = AnnotationController(app_config)
        controller.run()
    except SystemExit as e:
         log.info(f"Application exited: {e}")
    except ImportError as e: # Should be caught by the top-level try-except now
        log.critical(f"Missing Dependency Error during runtime: {e}", exc_info=True)
        print("\n--- Missing Dependency ---")
        print("Please ensure all required libraries are installed (see README.md).")
        sys.exit(1)
    except Exception as e:
        log.critical("An unexpected error occurred during application execution.", exc_info=True)
        print("\n--- An Unexpected Error Occurred ---")
        print(f"Error: {e}")
        print("Check the log messages above for details.")
        print("------------------------------------")
        sys.exit(1)
    finally:
        log.info("--- QuickMasker Application Finished ---")

if __name__ == "__main__":
    main()
