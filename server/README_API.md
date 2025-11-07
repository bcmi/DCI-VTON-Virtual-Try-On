# Virtual Try-On API (Backend)

### Endpoints
- **GET** `/healthz` → `{ "ok": true }`
- **POST** `/tryon`
  - `person`: file (image/jpeg or png)
  - `garment`: file (image/jpeg or png)
  - Returns: image/png

### Example
```bash
curl -o result.png -F person=@me.jpg -F garment=@shirt.jpg http://localhost:7860/tryon

