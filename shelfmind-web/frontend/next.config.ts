import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "kush5699-shelfmind-ai.hf.space",
      },
    ],
  },
};

export default nextConfig;
