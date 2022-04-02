import { Box, Button, Container, Text } from "@chakra-ui/react";
import { useRouter } from "next/router";

const Footer = () => {
  const router = useRouter();

  return (
    <Box w="100%" p={4} bg="pink.100">
      <Container textAlign="center">
        <Text mb={2} fontSize="sm">
          จัดทำโดยความร่วมมือระหว่าง ศูนย์โรคหัวใจ โรงพยาบาลศิริราช และ
          <br />
          ภาควิชาวิศวกรรมชีวการแพทย์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยมหิดล
        </Text>
        <Button
          variant="link"
          colorScheme="pink"
          fontSize="sm"
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
