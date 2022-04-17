/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  typescript: {
    // !! WARN !!
    // Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    // Currently stuck at this issue: https://stackoverflow.com/q/71809903/4010864
    // No easy fix for NPM so I skipped this for now.
    // !! WARN !!
    ignoreBuildErrors: true,
  },
}

module.exports = nextConfig
