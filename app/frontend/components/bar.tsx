import { Box, Flex, Icon } from "@chakra-ui/react";
import { FaCaretUp } from "react-icons/fa";

interface BarProp {
  value: number;
  min: number;
  max: number;
}

const Bar = ({ value, max }: BarProp) => {
  const percentage = Math.min((value * 100) / max, 99);

  return (
    <Flex my={4} position="relative">
      <Box h={4} w="100%" borderRadius="xl" background="green.400" />
      <Box
        h={4}
        w="16.7%"
        left="25%"
        background="green.200"
        position="absolute"
      />
      <Box
        h={4}
        w="41.7%"
        left="41.7%"
        background="red.300"
        position="absolute"
      />
      <Box
        h={4}
        w="16.6%"
        left="83.4%"
        background="red.500"
        position="absolute"
        borderRightRadius="xl"
      />
      <Box
        h={4}
        w="3px"
        position="absolute"
        top={0}
        left={`calc(${percentage}% - 1.5px)`}
        backgroundColor="gray.600"
        borderRadius="md"
      />
      <Box position="absolute" top={3} left={`calc(${percentage}% - 10px)`}>
        <Icon w="20px" h="20px" as={FaCaretUp} color="gray.600" />
      </Box>
    </Flex>
  );
};

export default Bar;
