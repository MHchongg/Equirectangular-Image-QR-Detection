# The overall flow

                equirectangular image
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
          Cubemap                Perspective
          6 views                 24 views
              │                     │
              └──────────┬──────────┘
                         ▼
                  Upscale image
                         │
                         ▼
              Create image variants
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Original       Grayscale       CLAHE
          │              │              │
          ├──────────────┼──────────────┤
          ▼              ▼
         Otsu      Adaptive Threshold
          │              │
          └──────────────┘
                         ▼
                  OpenCV QR detector
                         │
                         ▼
                    QR data

