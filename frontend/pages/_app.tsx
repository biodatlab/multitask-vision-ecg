import { ChakraProvider, extendTheme } from "@chakra-ui/react";
import "@fontsource/ibm-plex-sans-thai";
import type { AppProps } from "next/app";
import "../styles/globals.css";

const theme = extendTheme({
  fonts: {
    heading: '"IBM Plex Sans Thai", sans-serif',
    body: '"IBM Plex Sans Thai", sans-serif',
  },
});

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <ChakraProvider theme={theme}>
      <Component {...pageProps} />
    </ChakraProvider>
  );
}

export default MyApp;
