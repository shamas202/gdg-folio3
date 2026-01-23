# Interior Visual Search Frontend

A modern Next.js 16 + React 19 frontend for the Interior Visual Search application.

## Features

- **Visual Search**: Upload room images to find matching furniture products
- **Catalog Management**: Add products to the catalog with images and metadata
- **Professional UI**: Clean, modern design with SVG icons
- **Responsive**: Works seamlessly on desktop and mobile devices

## Tech Stack

- Next.js 16 (App Router)
- React 19
- TypeScript
- Tailwind CSS
- Lucide React (SVG icons)

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm, yarn, or pnpm

### Installation

1. Install dependencies:
```bash
npm install
# or
yarn install
# or
pnpm install
```

2. Set up environment variables (optional):
Create a `.env.local` file:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Run the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout with navigation
│   ├── page.tsx            # Home/search page
│   ├── results/
│   │   └── page.tsx        # Search results page
│   ├── catalog/
│   │   └── page.tsx        # Catalog management page
│   └── globals.css         # Global styles
├── components/
│   ├── Navigation.tsx      # Main navigation component
│   ├── SearchInterface.tsx # Image search interface
│   ├── SearchResults.tsx   # Results display component
│   └── CatalogManager.tsx  # Catalog management component
├── lib/
│   ├── api.ts              # API client functions
│   └── utils.ts            # Utility functions
└── package.json
```

## API Integration

The frontend communicates with the backend API at `http://localhost:8000` by default. Make sure the backend is running before using the frontend.

### Endpoints Used

- `GET /api/v1/health` - Health check
- `POST /api/v1/search` - Search for products
- `POST /api/v1/catalog/upsert` - Add product to catalog

## Building for Production

```bash
npm run build
npm start
```

## Design System

- **Primary Color**: Green (primary-600)
- **Accent Color**: Teal (accent-600)
- **Neutral Colors**: Gray scale
- **Icons**: Lucide React (SVG-based)
- **No blue/purple gradients**: Uses green/teal color scheme

