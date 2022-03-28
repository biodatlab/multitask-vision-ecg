import type { NextPage } from "next";
import Layout from "../components/layout";
import Dropzone from "../components/dropzone";
import { Divider, Flex, Heading, Stack } from "@chakra-ui/react";
import Prediction from "../components/prediction";
import { useState } from "react";

const Home: NextPage = () => {
  const [result, setResult] = useState(null);

  return (
    <Layout>
      <Stack pt={6} direction="column" textAlign={"center"} gap={2}>
        <Heading color="gray.500" as={"h1"}>
          12-lead ECG Classification
        </Heading>
        <Dropzone />
        {result && (
          <>
            <Divider />
            <Prediction normality={0.49} lvefgteq40={0.6} lveflw50={0.7} />
          </>
        )}
      </Stack>
    </Layout>
  );
};

export default Home;
