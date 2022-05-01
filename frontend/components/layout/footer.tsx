import { Box, Button, Container, Text } from "@chakra-ui/react";
import { useRouter } from "next/router";

const Footer = () => {
  const router = useRouter();

  return (
    <Box w="100%" pt={10} pb={8} bg="secondary.400">
      <Container textAlign="center">
        <Text mb={2} fontSize="sm" color="white">
          โมเดลนี้จัดทำโดยความร่วมมือระหว่างศูนย์โรคหัวใจ โรงพยาบาลศิริราช{" "}
          <Text as="span" display={{ base: "inline", md: "block" }}>
            และภาควิชาวิศวกรรมชีวการแพทย์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยมหิดล
          </Text>
        </Text>
        <Button
          variant="link"
          color="white"
          fontSize="sm"
          _hover={{
            textDecoration: "none",
          }}
          onClick={() => {
            router.push("/privacy-policy");
          }}
        >
          นโยบายความเป็นส่วนตัว
        </Button>
      </Container>
    </Box>
  );
};

export default Footer;
