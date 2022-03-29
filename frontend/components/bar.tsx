import { Flex, Box, Icon, Button } from "@chakra-ui/react";
import { FaCaretUp } from "react-icons/fa";

interface BarProp {
  value: number;
  min: number;
  max: number;
}

const Bar = ({ value, min, max }: BarProp) => {
  return (
    <Flex my={4} position="relative">
      <Box
        h={4}
        w="100%"
        borderRadius="xl"
        bgGradient="linear(to-r, pink.100, pink.200, red.400, red.500)"
      />
      <Box
        h={4}
        w="3px"
        position="absolute"
        top={0}
        left={`calc(${value}% - 1.5px)`}
        backgroundColor="gray.600"
        borderRadius="md"
      />
      <Box position="absolute" top={3} left={`calc(${value}% - 10px)`}>
        <Icon w="20px" h="20px" as={FaCaretUp} color="gray.600" />
      </Box>
    </Flex>
  );
};

export default Bar;
