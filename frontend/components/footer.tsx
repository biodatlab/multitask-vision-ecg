import { Box, Button, Container, Text } from "@chakra-ui/react";

const Footer = () => {
  return (
    <Box w="100%" p={4} bg="pink.100">
      <Container textAlign="center">
        <Text mb={2} fontSize="sm" color="gray.600">
          จัดทำโดยความร่วมมือระหว่างศูนย์โรคหัวใจ โรงพยาบาลศิริราช
          <br />
          และ ภาควิชาวิศวกรรมชีวการแพทย์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยมหิดล
        </Text>
        <Button variant="link" colorScheme="pink" fontSize="sm">
          นโยบายความเป็นส่วนตัว
        </Button>
      </Container>
    </Box>
  );
};

export default Footer;
