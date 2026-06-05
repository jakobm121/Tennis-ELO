from tennis_elo.normalize_matches import main as normalize_main
from tennis_elo.dedupe_matches import main as dedupe_main
from tennis_elo.build_elo import main as build_elo_main


def main():
    normalize_main()
    dedupe_main()
    build_elo_main()


if __name__ == "__main__":
    main()
