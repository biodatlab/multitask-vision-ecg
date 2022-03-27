import type { NextPage } from "next";
import Layout from "../components/layout";
import Dropzone from "../components/dropzone";
import { Flex } from "@chakra-ui/react";

const Home: NextPage = () => {
  return (
    <Layout>
      <Flex pt={8}>
        <Dropzone />
      </Flex>
    </Layout>
  );
};

export default Home;
