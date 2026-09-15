"""Convenience launcher for the point-source ray-tracing viewer."""

if __package__:
    from .point_source_gui import main
else:  # Supports ``python point_source_gui/start_gui.py`` from the repository root.
    from point_source_gui import main


if __name__ == "__main__":
    main()
