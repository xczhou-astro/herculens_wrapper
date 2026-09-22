# Source-point lens explorer

Local interactive viewer for `modelling_F150W_SIE/pixelated_hmc`.

```bash
cd "/Users/xczhou/Library/CloudStorage/GoogleDrive-xczhou95@gmail.com/My Drive/modelling/herculens_wrapper/source_point_lens_explorer"
python app.py
```

Open [http://127.0.0.1:5052](http://127.0.0.1:5052). The reconstructed source uses the Matplotlib `twilight` colormap. Click the source plane to add a point source. Select a source from the list to set its intrinsic brightness, or remove it. The second panel shows its lensed image positions; the third overlays their magnification-weighted point-spread markers on the original F150W image; the fourth overlays them on `image − fitted lens light` (the F150W ring residual).

The default input is the supplied SIE + external-shear result. To use another matching run:

```bash
python app.py --result-dir "/path/to/pixelated_hmc"
```

The source plane uses the saved uniform adaptive pixel grid. Image positions are solutions of the SIE + external-shear lens equation and are limited to the fitted image cutout.
