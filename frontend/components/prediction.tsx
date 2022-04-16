import { Box, Flex, Heading, Stack, Text, VStack } from "@chakra-ui/react";
import { prediction, predictionResult } from "../pages/ecg";

interface PredictionProps {
  predictionResult: predictionResult;
}

interface PredictionCardProps {
  data: prediction;
}

interface ProbabilityBarProps {
  probability: number;
  average: number;
}

const ProbabilityBar = ({ probability, average }: ProbabilityBarProps) => {
  const height = 24;
  const shift = height / 2 + 4;

  return (
    <Flex
      maxW={{ base: "100%", md: "77.5%" }}
      h={height}
      flex={1}
      position="relative"
    >
      {/* top labels */}
      <Flex
        width="100%"
        justifyContent="space-between"
        position="absolute"
        top={shift - 3 - 10 + 2}
      >
        <Text
          as="span"
          color="gray.800"
          fontSize="sm"
          ml={{ base: -1, md: -9 }}
        >
          ความเสี่ยงต่ำ
        </Text>
        <Text
          as="span"
          color="gray.800"
          fontSize="sm"
          mr={{ base: -1, md: -9 }}
        >
          ความเสี่ยงสูง
        </Text>
      </Flex>
      {/* bg bar */}
      <Box
        h={8}
        w="100%"
        position="absolute"
        top={shift - 4}
        backgroundColor="gray.100"
      />
      {/* gradient bar */}
      <Box
        h={8}
        w={`${probability}%`}
        position="absolute"
        top={shift - 4}
        bgGradient="linear(to-r, #E04586, #FF6651)"
        borderRightRadius="3xl"
      />
      {/* inner percent text */}
      <Box position="absolute" top="53px" right={`${102 - probability}%`}>
        <Text as="span" color="white" fontSize="md" fontWeight="semibold">
          {`${Math.round(probability)}%`}
        </Text>
      </Box>
      {/* average pole */}
      <Box
        h={10}
        w="2px"
        position="absolute"
        top={shift - 5 - 1}
        left={`calc(${average}% - 1px)`}
        backgroundColor="gray.800"
      />
      {/* average label */}
      <Box
        position="absolute"
        top={shift - 3 - 10 + 2}
        left={`calc(${average}% - 18px)`}
      >
        <Text as="span" color="gray.900" fontSize="xs">
          ค่าเฉลี่ย
        </Text>
      </Box>
      {/* bottom labels */}
      <Flex
        width="100%"
        justifyContent="space-between"
        position="absolute"
        bottom={0 - 2}
      >
        <Text as="span" color="gray.700" fontSize="xs">
          0
        </Text>
        <Text as="span" color="gray.700" fontSize="xs">
          25
        </Text>
        <Text as="span" color="gray.700" fontSize="xs">
          50
        </Text>
        <Text as="span" color="gray.700" fontSize="xs">
          75
        </Text>
        <Text as="span" color="gray.700" fontSize="xs">
          100%
        </Text>
      </Flex>
    </Flex>
  );
};

const PredictionCard = ({ data }: PredictionCardProps) => {
  const { title, description, risk_level, probability, average } = data;

  return (
    <Flex
      flex={1}
      w={{ base: "100%", md: "75%", lg: "60%" }}
      flexDirection="column"
      alignItems="center"
      borderRadius="2xl"
      boxShadow="lg"
      p={6}
    >
      <Flex w="100%" justifyContent="space-between" alignItems="center" mb={4}>
        <Box>
          <Heading as="h5" fontSize="2xl" color="secondary.400" mb={1}>
            {title}
          </Heading>
          <Text fontSize="sm" maxW="2xs">
            {description}
          </Text>
        </Box>
        <Box
          textAlign="center"
          backgroundColor="gray.50"
          borderRadius="lg"
          p={3}
        >
          <Text fontSize="sm" mb={3}>
            ระดับความเสี่ยง
          </Text>
          <Heading as="h5" fontSize="xl" color="primary.300" mb={1}>
            {risk_level}
          </Heading>
        </Box>
      </Flex>

      <Flex w="100%" justifyContent="center">
        <ProbabilityBar probability={probability} average={average} />
      </Flex>
    </Flex>
  );
};

const Prediction = ({ predictionResult }: PredictionProps) => {
  if (!Array.isArray(predictionResult)) {
    return null;
  }

  return (
    <Stack direction="column" gap={4} mt={10}>
      <Heading as="h4" fontSize="2xl" color="secondary.400" mb={6}>
        ผลทำนาย
      </Heading>
      <VStack gap={8}>
        {predictionResult.map((data) => (
          <PredictionCard key={data.title} data={data} />
        ))}
      </VStack>
    </Stack>
  );
};

export default Prediction;
