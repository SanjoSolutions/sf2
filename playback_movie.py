import retro
import os
from retro.scripts.playback_movie import main as retro_scripts_playback_movie_main

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
  retro.data.Integrations.add_custom_path(
    os.path.join(SCRIPT_DIR, "custom_integrations")
  )
  retro_scripts_playback_movie_main()


if __name__ == '__main__':
    main()
