import { SearchInterface } from "@/components/SearchInterface";

export default function Home() {
  return (
    <div className="w-full h-screen flex flex-col overflow-hidden">
      <div className="flex-shrink-0 px-6 py-4 border-b border-neutral-200">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-neutral-900 mb-2">
            Interior Visual Search
          </h1>
          <p className="text-sm text-neutral-600">
            Upload a room image to find exact product matches and visually similar alternatives
          </p>
        </div>
      </div>
      <div className="flex-1 overflow-hidden">
        <SearchInterface />
      </div>
    </div>
  );
}

