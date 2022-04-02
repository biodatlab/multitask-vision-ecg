import { ChakraProvider, extendTheme } from "@chakra-ui/react";
import "@fontsource/ibm-plex-sans-thai";
import type { AppProps } from "next/app";
import Head from "next/head";
import "../styles/globals.css";

const theme = extendTheme({
  fonts: {
    heading: '"IBM Plex Sans Thai", sans-serif',
    body: '"IBM Plex Sans Thai", sans-serif',
  },
});

function MyApp({ Component, pageProps }: AppProps) {
  // prettier-ignore
  return (
    <>
      <Head>
        <link rel="apple-touch-icon" sizes="180x180" href="/favicon/apple-touch-icon.png" />
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon/favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon/favicon-16x16.png" />
        <link rel="manifest" href="/favicon/site.webmanifest" />
        <link rel="mask-icon" href="/favicon/safari-pinned-tab.svg" color="#ed64a6" />
        <link rel="shortcut icon" href="/favicon/favicon.ico" />
        <meta name="msapplication-TileColor" content="#ed64a6" />
        <meta name="msapplication-config" content="/favicon/browserconfig.xml" />
        <meta name="theme-color" content="#ffffff" />
      </Head>
      <ChakraProvider theme={theme}>
        <Component {...pageProps} />
      </ChakraProvider>
    </>
  );
}

export default MyApp;
