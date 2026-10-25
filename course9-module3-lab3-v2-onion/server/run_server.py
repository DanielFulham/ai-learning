from application.container import initialise_server


def main() -> None:
    initialise_server().run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
