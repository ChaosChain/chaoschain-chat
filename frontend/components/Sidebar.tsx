import React from 'react';

const Sidebar: React.FC = () => {
  // Get data from environment variables
  const litepaperRepoUrl = process.env.NEXT_PUBLIC_LITEPAPER_SRC_URL || "";
  const repoName = litepaperRepoUrl.split('/').pop()?.replace('.git', '') || "N/A"; // Extract repo name
  const lastSynced = "Just now"; // Placeholder

  return (
    <aside className="w-64 flex-shrink-0 border-r border-border p-4 flex flex-col gap-4">
      <h2 className="text-lg font-semibold mb-2">Info</h2>
      
      <div>
        <h3 className="text-sm font-medium text-muted-foreground mb-1">Sources</h3>
        {/* Make the source a clickable link */}
        <a 
          href={litepaperRepoUrl} 
          target="_blank" 
          rel="noopener noreferrer" 
          className="text-sm break-words text-primary underline hover:no-underline"
        >
          {repoName}
        </a>
      </div>

      {/* Add other sidebar elements here if needed */}
    </aside>
  );
};

export default Sidebar; 